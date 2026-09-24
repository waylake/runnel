#include "runnel/quantized_qmv_probe.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef RUNNEL_DEFAULT_QUANTIZED_QMV_SHADER
#define RUNNEL_DEFAULT_QUANTIZED_QMV_SHADER \
  "shaders/quantized_qmv.metal"
#endif

namespace runnel::metal {
namespace {

using Clock = std::chrono::steady_clock;

struct alignas(16) QmvParams {
  std::uint32_t bits;
  std::uint32_t reduction_size;
  std::uint32_t output_size;
  std::uint32_t group_count;
  std::uint32_t quant_group_size;
  std::uint32_t weight_bank_output_size;
  std::uint32_t rows_per_expert;
};

struct DatasetHeader {
  char magic[8]{'R', 'N', 'L', 'Q', 'M', 'V', '0', '2'};
  std::uint32_t version{2};
  std::uint32_t bits{};
  std::uint32_t output_size{};
  std::uint32_t weight_bank_output_size{};
  std::uint32_t rows_per_expert{};
  std::uint32_t reduction_size{};
  std::uint32_t quant_group_size{};
  std::uint32_t selected_expert_count{};
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
  static_assert(sizeof(bits) == sizeof(value));
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

class QuantizedQmvProbe::Impl {
 public:
  Impl(
      QuantizationBits bits,
      std::size_t output_size,
      std::size_t reduction_size,
      std::size_t quant_group_size,
      std::size_t weight_bank_output_size,
      std::size_t rows_per_expert,
      std::vector<std::uint32_t> selected_experts)
      : bits_(bits),
        output_size_(output_size),
        reduction_size_(reduction_size),
        quant_group_size_(quant_group_size),
        weight_bank_output_size_(
            weight_bank_output_size == 0 ? output_size : weight_bank_output_size),
        rows_per_expert_(rows_per_expert == 0 ? output_size : rows_per_expert),
        selected_experts_(std::move(selected_experts)),
        values_per_pack_(bits == QuantizationBits::q4 ? 8 : 4) {
    if (output_size == 0 || reduction_size == 0 ||
        reduction_size > 2048) {
      fail("QMV probe requires 0 < reduction size <= 2048");
    }
    if (reduction_size % values_per_pack_ != 0 ||
        reduction_size % quant_group_size_ != 0 ||
        output_size > std::numeric_limits<std::uint32_t>::max() ||
        weight_bank_output_size_ > std::numeric_limits<std::uint32_t>::max()) {
      fail("QMV probe shape is incompatible with the packed affine layout");
    }
    if (selected_experts_.empty()) {
      selected_experts_.push_back(0);
    }
    if (rows_per_expert_ == 0 ||
        weight_bank_output_size_ % rows_per_expert_ != 0 ||
        output_size_ != selected_experts_.size() * rows_per_expert_) {
      fail("QMV expert bank and active output shape do not agree");
    }
    const std::size_t expert_count = weight_bank_output_size_ / rows_per_expert_;
    if (std::any_of(
            selected_experts_.begin(), selected_experts_.end(),
            [&](std::uint32_t index) {
              return static_cast<std::size_t>(index) >= expert_count;
            })) {
      fail("QMV selected expert is outside the weight bank");
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
        stringWithContentsOfFile:@(RUNNEL_DEFAULT_QUANTIZED_QMV_SHADER)
                       encoding:NSUTF8StringEncoding
                          error:&error];
    if (source == nil) {
      fail(ns_error(error, "could not read the quantized QMV shader"));
    }
    id<MTLLibrary> library = [device_ newLibraryWithSource:source
                                                     options:nil
                                                       error:&error];
    if (library == nil) {
      fail(ns_error(error, "could not compile the quantized QMV shader"));
    }

    id<MTLFunction> function =
        [library newFunctionWithName:@"qmv_weight_major"];
    if (function == nil) {
      fail("quantized QMV shader is missing qmv_weight_major");
    }
    pipeline_ = [device_ newComputePipelineStateWithFunction:function
                                                       error:&error];
    if (pipeline_ == nil) {
      fail(ns_error(error, "could not create the quantized QMV pipeline"));
    }

    build_data();
    allocate_buffers();
  }

  QmvMeasurement measure(
      std::size_t threadgroups,
      std::size_t threads_per_threadgroup,
      std::uint32_t warmups,
      std::uint32_t trials) const {
    if (threadgroups == 0 || threads_per_threadgroup == 0 ||
        threads_per_threadgroup % 32 != 0) {
      fail("QMV threadgroups and 32-wide threadgroup sizes must be positive");
    }
    if (threads_per_threadgroup > pipeline_.maxTotalThreadsPerThreadgroup) {
      fail("QMV threadgroup exceeds the device pipeline limit");
    }

    const std::size_t simdgroups = threads_per_threadgroup / 32;
    const std::size_t denominator = threadgroups * simdgroups;
    if (output_size_ % denominator != 0) {
      fail("QMV output size must divide evenly across persistent workgroups");
    }
    const std::size_t rows_per_simdgroup = output_size_ / denominator;
    if (rows_per_simdgroup < 4 || rows_per_simdgroup % 4 != 0) {
      fail("QMV persistent schedule requires four output rows per SIMD group");
    }
    if (weight_bank_output_size_ != output_size_ &&
        (rows_per_simdgroup > rows_per_expert_ ||
         rows_per_expert_ % rows_per_simdgroup != 0)) {
      fail("QMV SIMD tile must not cross an expert boundary");
    }

    const QmvParams params{
        static_cast<std::uint32_t>(bits_ == QuantizationBits::q4 ? 4 : 8),
        static_cast<std::uint32_t>(reduction_size_),
        static_cast<std::uint32_t>(output_size_),
        static_cast<std::uint32_t>(threadgroups),
        static_cast<std::uint32_t>(quant_group_size_),
        static_cast<std::uint32_t>(weight_bank_output_size_),
        static_cast<std::uint32_t>(rows_per_expert_),
    };
    std::memcpy(params_buffer_.contents, &params, sizeof(params));

    const auto run_once = [&]() {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pipeline_];
      [encoder setBuffer:weights_buffer_ offset:0 atIndex:0];
      [encoder setBuffer:scales_buffer_ offset:0 atIndex:1];
      [encoder setBuffer:biases_buffer_ offset:0 atIndex:2];
      [encoder setBuffer:x_buffer_ offset:0 atIndex:3];
      [encoder setBuffer:y_buffer_ offset:0 atIndex:4];
      [encoder setBuffer:params_buffer_ offset:0 atIndex:5];
      [encoder setBuffer:indices_buffer_ offset:0 atIndex:6];
      [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
      [encoder endEncoding];
      [command_buffer commit];

      const auto started = Clock::now();
      [command_buffer waitUntilCompleted];
      const auto ended = Clock::now();
      if (command_buffer.error != nil) {
        fail(ns_error(command_buffer.error, "quantized QMV kernel failed"));
      }
      if (command_buffer.GPUStartTime == 0 || command_buffer.GPUEndTime == 0) {
        fail("Metal did not provide GPU timing for quantized QMV");
      }
      return std::pair<double, double>{
          std::chrono::duration<double, std::micro>(ended - started).count(),
          (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1e6,
      };
    };

    for (std::uint32_t index = 0; index < warmups; ++index) {
      static_cast<void>(run_once());
    }

    QmvMeasurement measurement;
    measurement.output_size = output_size_;
    measurement.reduction_size = reduction_size_;
    measurement.threadgroups = threadgroups;
    measurement.threads_per_threadgroup = threads_per_threadgroup;
    measurement.weight_bytes = weights_.size() * sizeof(std::uint32_t) /
        selected_experts_.size();
    measurement.wall_micros.reserve(trials);
    measurement.gpu_micros.reserve(trials);
    for (std::uint32_t index = 0; index < trials; ++index) {
      auto [wall, gpu] = run_once();
      measurement.wall_micros.push_back(wall);
      measurement.gpu_micros.push_back(gpu);
    }

    const auto* actual = static_cast<const float*>(y_buffer_.contents);
    measurement.output_checksum = 0.0;
    measurement.reference_checksum = 0.0;
    double squared_error_sum = 0.0;
    const std::size_t preview_count = std::min<std::size_t>(8, output_size_);
    measurement.first_actual.reserve(preview_count);
    measurement.first_reference.reserve(preview_count);
    for (std::size_t row = 0; row < output_size_; ++row) {
      if (!std::isfinite(actual[row])) {
        fail("quantized QMV produced a non-finite output");
      }
      measurement.output_checksum += actual[row];
      measurement.reference_checksum += expected_[row];
      measurement.max_reference_magnitude = std::max(
          measurement.max_reference_magnitude,
          std::abs(static_cast<double>(expected_[row])));
      const double error = std::abs(
          static_cast<double>(actual[row]) - expected_[row]);
      squared_error_sum += error * error;
      measurement.max_absolute_error =
          std::max(measurement.max_absolute_error, error);
      measurement.max_relative_error = std::max(
          measurement.max_relative_error,
          error / std::max(1e-6, std::abs(static_cast<double>(expected_[row]))));
      if (row < preview_count) {
        measurement.first_actual.push_back(actual[row]);
        measurement.first_reference.push_back(expected_[row]);
      }
    }
    measurement.rmse = std::sqrt(squared_error_sum / output_size_);
    return measurement;
  }

  void write_dataset(const std::string& path) const {
    const std::filesystem::path output_path(path);
    if (!output_path.parent_path().empty()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream stream(output_path, std::ios::binary);
    if (!stream) {
      fail("could not open QMV dataset output: " + path);
    }

    DatasetHeader header;
    header.bits = bits_ == QuantizationBits::q4 ? 4 : 8;
    header.output_size = static_cast<std::uint32_t>(output_size_);
    header.weight_bank_output_size =
        static_cast<std::uint32_t>(weight_bank_output_size_);
    header.rows_per_expert = static_cast<std::uint32_t>(rows_per_expert_);
    header.reduction_size = static_cast<std::uint32_t>(reduction_size_);
    header.quant_group_size = static_cast<std::uint32_t>(quant_group_size_);
    header.selected_expert_count =
        static_cast<std::uint32_t>(selected_experts_.size());
    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.write(
        reinterpret_cast<const char*>(weights_.data()),
        static_cast<std::streamsize>(weights_.size() * sizeof(std::uint32_t)));
    stream.write(
        reinterpret_cast<const char*>(scales_.data()),
        static_cast<std::streamsize>(scales_.size() * sizeof(std::uint16_t)));
    stream.write(
        reinterpret_cast<const char*>(biases_.data()),
        static_cast<std::streamsize>(biases_.size() * sizeof(std::uint16_t)));
    stream.write(
        reinterpret_cast<const char*>(x_.data()),
        static_cast<std::streamsize>(x_.size() * sizeof(std::uint16_t)));
    stream.write(
        reinterpret_cast<const char*>(selected_experts_.data()),
        static_cast<std::streamsize>(
            selected_experts_.size() * sizeof(std::uint32_t)));
    if (!stream) {
      fail("failed while writing QMV dataset: " + path);
    }
  }

 private:
  std::size_t map_weight_row(std::size_t active_row) const {
    if (weight_bank_output_size_ == output_size_ && selected_experts_.size() == 1) {
      return active_row;
    }
    const std::size_t local_expert = active_row / rows_per_expert_;
    const std::size_t row_in_expert = active_row % rows_per_expert_;
    return static_cast<std::size_t>(selected_experts_.at(local_expert)) *
        rows_per_expert_ + row_in_expert;
  }

  void build_data() {
    const std::size_t packed_stride = reduction_size_ / values_per_pack_;
    weights_.resize(weight_bank_output_size_ * packed_stride);
    const std::size_t quant_groups = reduction_size_ / quant_group_size_;
    const std::size_t parameter_count = weight_bank_output_size_ * quant_groups;
    scales_.resize(parameter_count);
    biases_.resize(parameter_count);
    x_.resize(reduction_size_);
    expected_.resize(output_size_);

    x_values_.resize(reduction_size_);
    Random random(bits_ == QuantizationBits::q4
                      ? 0x4A17C0DEULL
                      : 0x8A17C0DEULL);
    for (std::size_t index = 0; index < reduction_size_; ++index) {
      const float value = random.uniform(-1.0f, 1.0f);
      x_.at(index) = float_to_bf16(value);
      x_values_.at(index) = bf16_to_float(x_.at(index));
    }

    const float scale_low = bits_ == QuantizationBits::q4 ? 0.002f : 0.00005f;
    const float scale_high = bits_ == QuantizationBits::q4 ? 0.008f : 0.00015f;
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
      scales_.at(parameter) = float_to_bf16(
          random.uniform(scale_low, scale_high));
      biases_.at(parameter) = float_to_bf16(random.uniform(-0.0005f, 0.0005f));
    }

    const unsigned maximum_quantized =
        bits_ == QuantizationBits::q4 ? 0x0FU : 0xFFU;
    for (std::size_t row = 0; row < weight_bank_output_size_; ++row) {
      for (std::size_t packed = 0; packed < packed_stride; ++packed) {
        std::uint32_t value = 0;
        for (std::size_t byte = 0; byte < 4; ++byte) {
          const std::uint32_t quantized =
              random.next_u32() & maximum_quantized;
          value |= quantized << (byte * 8);
        }
        weights_.at(row * packed_stride + packed) = value;
      }
    }

    for (std::size_t row = 0; row < output_size_; ++row) {
      const std::size_t weight_row = map_weight_row(row);
      float result = 0.0f;
      for (std::size_t group = 0; group < quant_groups; ++group) {
        const float scale =
            bf16_to_float(scales_.at(weight_row * quant_groups + group));
        const float bias =
            bf16_to_float(biases_.at(weight_row * quant_groups + group));
        float weighted_sum = 0.0f;
        float x_sum = 0.0f;
        for (std::size_t in_group = 0; in_group < quant_group_size_; ++in_group) {
          const std::size_t reduction = group * quant_group_size_ + in_group;
          const std::size_t packed = packed_stride * weight_row + reduction / values_per_pack_;
          const std::size_t shift = bits_ == QuantizationBits::q4
              ? 4 * (reduction % values_per_pack_)
              : 8 * (reduction % values_per_pack_);
          const std::uint32_t quantized =
              (weights_.at(packed) >> shift) & maximum_quantized;
          weighted_sum +=
              static_cast<float>(quantized) * scale * x_values_.at(reduction);
          x_sum += x_values_.at(reduction);
        }
        result += weighted_sum + x_sum * bias;
      }
      expected_.at(row) = result;
    }
  }

  void allocate_buffers() {
    const std::size_t parameter_bytes = scales_.size() * sizeof(std::uint16_t);
    weights_buffer_ = [device_ newBufferWithBytes:weights_.data()
                                            length:weights_.size() * sizeof(std::uint32_t)
                                           options:MTLResourceStorageModeShared];
    scales_buffer_ = [device_ newBufferWithBytes:scales_.data()
                                           length:parameter_bytes
                                          options:MTLResourceStorageModeShared];
    biases_buffer_ = [device_ newBufferWithBytes:biases_.data()
                                           length:parameter_bytes
                                          options:MTLResourceStorageModeShared];
    x_buffer_ = [device_ newBufferWithBytes:x_.data()
                                      length:x_.size() * sizeof(std::uint16_t)
                                     options:MTLResourceStorageModeShared];
    y_buffer_ = [device_ newBufferWithLength:output_size_ * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    params_buffer_ = [device_ newBufferWithLength:sizeof(QmvParams)
                                          options:MTLResourceStorageModeShared];
    indices_buffer_ = [device_
        newBufferWithBytes:selected_experts_.data()
                     length:selected_experts_.size() * sizeof(std::uint32_t)
                    options:MTLResourceStorageModeShared];
    if (weights_buffer_ == nil || scales_buffer_ == nil ||
        biases_buffer_ == nil || x_buffer_ == nil || y_buffer_ == nil ||
        params_buffer_ == nil || indices_buffer_ == nil) {
      fail("could not allocate quantized QMV buffers");
    }
  }

  QuantizationBits bits_;
  std::size_t output_size_;
  std::size_t reduction_size_;
  std::size_t quant_group_size_;
  std::size_t weight_bank_output_size_;
  std::size_t rows_per_expert_;
  std::vector<std::uint32_t> selected_experts_;
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
  __strong id<MTLBuffer> indices_buffer_{};
};

QuantizedQmvProbe::QuantizedQmvProbe(
    QuantizationBits bits,
    std::size_t output_size,
    std::size_t reduction_size,
    std::size_t quant_group_size,
    std::size_t weight_bank_output_size,
    std::size_t rows_per_expert,
    std::vector<std::uint32_t> selected_experts)
    : impl_(std::make_unique<Impl>(
          bits,
          output_size,
          reduction_size,
          quant_group_size,
          weight_bank_output_size,
          rows_per_expert,
          std::move(selected_experts))) {}

QuantizedQmvProbe::~QuantizedQmvProbe() = default;
QuantizedQmvProbe::QuantizedQmvProbe(QuantizedQmvProbe&&) noexcept = default;
QuantizedQmvProbe& QuantizedQmvProbe::operator=(QuantizedQmvProbe&&) noexcept =
    default;

QmvMeasurement QuantizedQmvProbe::measure(
    std::size_t threadgroups,
    std::size_t threads_per_threadgroup,
    std::uint32_t warmups,
    std::uint32_t trials) const {
  return impl_->measure(
      threadgroups, threads_per_threadgroup, warmups, trials);
}

void QuantizedQmvProbe::write_dataset(const std::string& path) const {
  impl_->write_dataset(path);
}

}  // namespace runnel::metal
