#include "runnel/quantized_qmm_probe.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef RUNNEL_DEFAULT_QUANTIZED_QMM_SHADER
#define RUNNEL_DEFAULT_QUANTIZED_QMM_SHADER "shaders/quantized_qmm.metal"
#endif

namespace runnel::metal {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kTokenTile = 8;
constexpr std::size_t kOutputTile = 32;
constexpr std::size_t kQuantGroupSize = 64;
constexpr std::size_t kReductionTile = 256;

struct alignas(16) QmmParams {
  std::uint32_t bits;
  std::uint32_t token_count;
  std::uint32_t output_count;
  std::uint32_t reduction_size;
  std::uint32_t quant_group_count;
  std::uint32_t token_tile;
  std::uint32_t output_tile;
  std::uint32_t reduction_tile;
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

std::string ns_error(NSError* error, const std::string& context) {
  std::ostringstream stream;
  stream << context;
  if (error != nil) {
    stream << ": " << error.localizedDescription.UTF8String;
  }
  return stream.str();
}

class Random {
 public:
  explicit Random(std::uint64_t seed) : state_(seed) {}

  std::uint32_t next_u32() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return static_cast<std::uint32_t>(state_ >> 32);
  }

  float uniform(float low, float high) {
    const double unit = static_cast<double>(next_u32()) / 0xFFFFFFFFU;
    return low + static_cast<float>(unit) * (high - low);
  }

 private:
  std::uint64_t state_;
};

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if (((bits >> 23) & 0xFFU) == 0xFFU) {
    return static_cast<std::uint16_t>(bits >> 16);
  }
  const std::uint32_t rounding = 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

}  // namespace

class QuantizedQmmProbe::Impl {
 public:
  Impl(
      QuantizationBits bits,
      std::size_t token_count,
      std::size_t output_count,
      std::size_t reduction_size,
      std::size_t quant_group_size)
      : bits_(bits),
        token_count_(token_count),
        output_count_(output_count),
        reduction_size_(reduction_size),
        quant_group_size_(quant_group_size),
        values_per_pack_(bits == QuantizationBits::q4 ? 8 : 4) {
    if (token_count_ == 0 || output_count_ == 0 || reduction_size_ == 0 ||
        token_count_ % kTokenTile != 0 || output_count_ % kOutputTile != 0 ||
        reduction_size_ % kReductionTile != 0 ||
        quant_group_size_ != kQuantGroupSize) {
      fail("qmm probe shape is incompatible with the 4D tile contract");
    }
    if (reduction_size_ % values_per_pack_ != 0 ||
        reduction_size_ % quant_group_size_ != 0 ||
        token_count_ > std::numeric_limits<std::uint32_t>::max() ||
        output_count_ > std::numeric_limits<std::uint32_t>::max()) {
      fail("qmm probe shape is incompatible with packed affine storage");
    }

    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) {
      fail("Metal is unavailable on this machine");
    }
    queue_ = [device_ newCommandQueue];
    if (queue_ == nil) {
      fail("could not create a Metal command queue");
    }

    NSError* error = nil;
    NSString* source = [NSString
        stringWithContentsOfFile:@(RUNNEL_DEFAULT_QUANTIZED_QMM_SHADER)
                       encoding:NSUTF8StringEncoding
                          error:&error];
    if (source == nil) {
      fail(ns_error(error, "could not read the quantized QMM shader"));
    }
    id<MTLLibrary> library = [device_ newLibraryWithSource:source
                                                     options:nil
                                                       error:&error];
    if (library == nil) {
      fail(ns_error(error, "could not compile the quantized QMM shader"));
    }
    id<MTLFunction> function = [library newFunctionWithName:@"qmm_4d_tile"];
    if (function == nil) {
      fail("quantized QMM shader is missing qmm_4d_tile");
    }
    pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
    if (pipeline_ == nil) {
      fail(ns_error(error, "could not create the quantized QMM pipeline"));
    }

    build_data();
    allocate_buffers();
  }

  QmmMeasurement measure(std::uint32_t warmups, std::uint32_t trials) const {
    if (trials == 0) {
      fail("qmm trials must be positive");
    }
    const std::size_t output_tiles = output_count_ / kOutputTile;
    const std::size_t token_tiles = token_count_ / kTokenTile;
    const QmmParams params{
        static_cast<std::uint32_t>(bits_ == QuantizationBits::q4 ? 4 : 8),
        static_cast<std::uint32_t>(token_count_),
        static_cast<std::uint32_t>(output_count_),
        static_cast<std::uint32_t>(reduction_size_),
        static_cast<std::uint32_t>(reduction_size_ / quant_group_size_),
        static_cast<std::uint32_t>(kTokenTile),
        static_cast<std::uint32_t>(kOutputTile),
        static_cast<std::uint32_t>(kReductionTile),
    };
    std::memcpy(params_buffer_.contents, &params, sizeof(params));

    const auto run_once = [&]() {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pipeline_];
      [encoder setBuffer:weights_buffer_ offset:0 atIndex:0];
      [encoder setBuffer:scales_buffer_ offset:0 atIndex:1];
      [encoder setBuffer:biases_buffer_ offset:0 atIndex:2];
      [encoder setBuffer:x_buffer_ offset:0 atIndex:3];
      [encoder setBuffer:y_buffer_ offset:0 atIndex:4];
      [encoder setBuffer:params_buffer_ offset:0 atIndex:5];
      [encoder dispatchThreadgroups:MTLSizeMake(output_tiles, token_tiles, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command_buffer commit];
      const auto started = Clock::now();
      [command_buffer waitUntilCompleted];
      const auto ended = Clock::now();
      if (command_buffer.error != nil) {
        fail(ns_error(command_buffer.error, "quantized QMM kernel failed"));
      }
      if (command_buffer.GPUStartTime == 0 || command_buffer.GPUEndTime == 0) {
        fail("Metal did not provide GPU timing for quantized QMM");
      }
      return std::pair<double, double>{
          std::chrono::duration<double, std::micro>(ended - started).count(),
          (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1e6,
      };
    };

    for (std::uint32_t index = 0; index < warmups; ++index) {
      static_cast<void>(run_once());
    }
    QmmMeasurement measurement;
    measurement.token_count = token_count_;
    measurement.output_count = output_count_;
    measurement.reduction_size = reduction_size_;
    measurement.threadgroups = output_tiles * token_tiles;
    measurement.weight_bytes = weights_.size() * sizeof(std::uint32_t);
    measurement.wall_micros.reserve(trials);
    measurement.gpu_micros.reserve(trials);
    for (std::uint32_t index = 0; index < trials; ++index) {
      auto [wall, gpu] = run_once();
      measurement.wall_micros.push_back(wall);
      measurement.gpu_micros.push_back(gpu);
    }

    const auto* actual = static_cast<const float*>(y_buffer_.contents);
    double squared_error = 0.0;
    for (std::size_t token = 0; token < token_count_; ++token) {
      for (std::size_t output = 0; output < output_count_; ++output) {
        const std::size_t index = token * output_count_ + output;
        if (!std::isfinite(actual[index])) {
          fail("quantized QMM produced a non-finite output");
        }
        measurement.output_checksum += actual[index];
        measurement.reference_checksum += expected_[index];
        const double error =
            static_cast<double>(actual[index]) - expected_[index];
        measurement.max_absolute_error =
            std::max(measurement.max_absolute_error, std::abs(error));
        squared_error += error * error;
      }
    }
    measurement.rmse = std::sqrt(
        squared_error / static_cast<double>(token_count_ * output_count_));
    return measurement;
  }

 private:
  void build_data() {
    const std::size_t packed_stride = reduction_size_ / values_per_pack_;
    const std::size_t quant_groups = reduction_size_ / quant_group_size_;
    weights_.resize(output_count_ * packed_stride);
    scales_.resize(output_count_ * quant_groups);
    biases_.resize(output_count_ * quant_groups);
    x_.resize(token_count_ * reduction_size_);
    x_values_.resize(x_.size());
    expected_.resize(token_count_ * output_count_);

    Random random(bits_ == QuantizationBits::q4
                      ? 0x4D4D4D01ULL
                      : 0x4D4D4D02ULL);
    const float scale_low = bits_ == QuantizationBits::q4 ? 0.002f : 0.00005f;
    const float scale_high = bits_ == QuantizationBits::q4 ? 0.008f : 0.00015f;
    const std::uint32_t maximum =
        bits_ == QuantizationBits::q4 ? 0x0FU : 0xFFU;
    for (std::size_t index = 0; index < x_.size(); ++index) {
      const float value = random.uniform(-1.0f, 1.0f);
      x_.at(index) = float_to_bf16(value);
      x_values_.at(index) = bf16_to_float(x_.at(index));
    }
    for (std::size_t row = 0; row < output_count_; ++row) {
      for (std::size_t group = 0; group < quant_groups; ++group) {
        scales_.at(row * quant_groups + group) = float_to_bf16(
            random.uniform(scale_low, scale_high));
        biases_.at(row * quant_groups + group) =
            float_to_bf16(random.uniform(-0.0005f, 0.0005f));
      }
      for (std::size_t packed = 0; packed < packed_stride; ++packed) {
        std::uint32_t value = 0;
        for (std::size_t byte = 0; byte < 4; ++byte) {
          value |= (random.next_u32() & maximum) << (byte * 8);
        }
        weights_.at(row * packed_stride + packed) = value;
      }
    }

    for (std::size_t token = 0; token < token_count_; ++token) {
      for (std::size_t output = 0; output < output_count_; ++output) {
        float result = 0.0f;
        for (std::size_t group = 0; group < quant_groups; ++group) {
          const float scale = bf16_to_float(scales_.at(output * quant_groups + group));
          const float bias = bf16_to_float(biases_.at(output * quant_groups + group));
          float weighted = 0.0f;
          float input_sum = 0.0f;
          for (std::size_t in_group = 0; in_group < quant_group_size_; ++in_group) {
            const std::size_t reduction = group * quant_group_size_ + in_group;
            const std::size_t packed =
                output * packed_stride + reduction / values_per_pack_;
            const std::size_t shift = bits_ == QuantizationBits::q4
                ? 4 * (reduction % values_per_pack_)
                : 8 * (reduction % values_per_pack_);
            const std::uint32_t quantized =
                (weights_.at(packed) >> shift) & maximum;
            const float input = x_values_.at(token * reduction_size_ + reduction);
            weighted += static_cast<float>(quantized) * input;
            input_sum += input;
          }
          result += weighted * scale + input_sum * bias;
        }
        expected_.at(token * output_count_ + output) = result;
      }
    }
  }

  void allocate_buffers() {
    weights_buffer_ = [device_
        newBufferWithBytes:weights_.data()
                     length:weights_.size() * sizeof(std::uint32_t)
                    options:MTLResourceStorageModeShared];
    scales_buffer_ = [device_
        newBufferWithBytes:scales_.data()
                     length:scales_.size() * sizeof(std::uint16_t)
                    options:MTLResourceStorageModeShared];
    biases_buffer_ = [device_
        newBufferWithBytes:biases_.data()
                     length:biases_.size() * sizeof(std::uint16_t)
                    options:MTLResourceStorageModeShared];
    x_buffer_ = [device_ newBufferWithBytes:x_.data()
                                      length:x_.size() * sizeof(std::uint16_t)
                                     options:MTLResourceStorageModeShared];
    y_buffer_ = [device_ newBufferWithLength:expected_.size() * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    params_buffer_ = [device_ newBufferWithLength:sizeof(QmmParams)
                                          options:MTLResourceStorageModeShared];
    if (weights_buffer_ == nil || scales_buffer_ == nil || biases_buffer_ == nil ||
        x_buffer_ == nil || y_buffer_ == nil || params_buffer_ == nil) {
      fail("could not allocate quantized QMM buffers");
    }
  }

  QuantizationBits bits_;
  std::size_t token_count_;
  std::size_t output_count_;
  std::size_t reduction_size_;
  std::size_t quant_group_size_;
  std::size_t values_per_pack_;
  std::vector<std::uint32_t> weights_;
  std::vector<std::uint16_t> scales_;
  std::vector<std::uint16_t> biases_;
  std::vector<std::uint16_t> x_;
  std::vector<float> x_values_;
  std::vector<float> expected_;
  __strong id<MTLDevice> device_{};
  __strong id<MTLCommandQueue> queue_{};
  __strong id<MTLComputePipelineState> pipeline_{};
  __strong id<MTLBuffer> weights_buffer_{};
  __strong id<MTLBuffer> scales_buffer_{};
  __strong id<MTLBuffer> biases_buffer_{};
  __strong id<MTLBuffer> x_buffer_{};
  __strong id<MTLBuffer> y_buffer_{};
  __strong id<MTLBuffer> params_buffer_{};
};

QuantizedQmmProbe::QuantizedQmmProbe(
    QuantizationBits bits,
    std::size_t token_count,
    std::size_t output_count,
    std::size_t reduction_size,
    std::size_t quant_group_size)
    : impl_(std::make_unique<Impl>(
          bits,
          token_count,
          output_count,
          reduction_size,
          quant_group_size)) {}

QuantizedQmmProbe::~QuantizedQmmProbe() = default;
QuantizedQmmProbe::QuantizedQmmProbe(QuantizedQmmProbe&&) noexcept = default;
QuantizedQmmProbe& QuantizedQmmProbe::operator=(QuantizedQmmProbe&&) noexcept =
    default;

QmmMeasurement QuantizedQmmProbe::measure(
    std::uint32_t warmups,
    std::uint32_t trials) const {
  return impl_->measure(warmups, trials);
}

}  // namespace runnel::metal
