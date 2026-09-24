#include "runnel/fused_moe_probe.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef RUNNEL_DEFAULT_FUSED_MOE_SHADER
#define RUNNEL_DEFAULT_FUSED_MOE_SHADER "shaders/fused_moe.metal"
#endif

namespace runnel::metal {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kHiddenSize = 2048;
constexpr std::size_t kIntermediateSize = 512;
constexpr std::size_t kGateUpRowsPerExpert = 2 * kIntermediateSize;
constexpr std::size_t kDownRowsPerExpert = kHiddenSize;
constexpr std::size_t kSelectedExperts = 8;
constexpr std::size_t kQuantGroupSize = 64;

struct alignas(16) FusedMoeParams {
  std::uint32_t bits;
  std::uint32_t group_count;
  std::uint32_t expert_count;
  std::uint32_t simdgroups_per_expert;
  std::uint32_t channels_per_simdgroup;
  std::uint32_t hidden_size;
  std::uint32_t intermediate_size;
  std::uint32_t debug_stage;
};

struct FusedDatasetHeader {
  char magic[8]{'R', 'N', 'L', 'M', 'O', 'E', '0', '1'};
  std::uint32_t version{1};
  std::uint32_t bits{};
  std::uint32_t total_experts{};
  std::uint32_t selected_expert_count{};
  std::uint32_t gate_up_rows_per_expert{};
  std::uint32_t down_rows_per_expert{};
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
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

class FusedMoeProbe::Impl {
 public:
  Impl(
      QuantizationBits bits,
      std::size_t total_experts,
      std::vector<std::uint32_t> selected_experts,
      std::uint32_t debug_stage,
      std::string dataset_path)
      : bits_(bits),
        total_experts_(total_experts),
        selected_experts_(std::move(selected_experts)),
        values_per_pack_(bits == QuantizationBits::q4 ? 8 : 4),
        debug_stage_(debug_stage),
        dataset_path_(std::move(dataset_path)) {
    if (debug_stage > 2) {
      fail("unsupported fused MoE debug stage");
    }
    if (total_experts_ == 0 || selected_experts_.size() != kSelectedExperts) {
      fail("fused MoE probe requires eight selected experts");
    }
    std::vector<std::uint32_t> sorted = selected_experts_;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      fail("fused MoE selected experts must be unique");
    }
    if (static_cast<std::size_t>(sorted.back()) >= total_experts_) {
      fail("fused MoE selected expert is outside the synthetic bank");
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
        stringWithContentsOfFile:@(RUNNEL_DEFAULT_FUSED_MOE_SHADER)
                       encoding:NSUTF8StringEncoding
                          error:&error];
    if (source == nil) {
      fail(ns_error(error, "could not read the fused MoE shader"));
    }
    id<MTLLibrary> library = [device_ newLibraryWithSource:source
                                                     options:nil
                                                       error:&error];
    if (library == nil) {
      fail(ns_error(error, "could not compile the fused MoE shader"));
    }
    id<MTLFunction> layer_function =
        [library newFunctionWithName:@"fused_moe_layer"];
    id<MTLFunction> combine_function =
        [library newFunctionWithName:@"combine_moe"];
    id<MTLFunction> gate_up_function =
        [library newFunctionWithName:@"gate_up_swiglu"];
    id<MTLFunction> down_function =
        [library newFunctionWithName:@"down_router"];
    id<MTLFunction> combine_router_function =
        [library newFunctionWithName:@"combine_router"];
    if (layer_function == nil || combine_function == nil ||
        gate_up_function == nil || down_function == nil ||
        combine_router_function == nil) {
      fail("fused MoE shader is missing a required kernel");
    }
    layer_pipeline_ =
        [device_ newComputePipelineStateWithFunction:layer_function error:&error];
    if (layer_pipeline_ == nil) {
      fail(ns_error(error, "could not create the fused MoE layer pipeline"));
    }
    combine_pipeline_ = [device_ newComputePipelineStateWithFunction:combine_function
                                                              error:&error];
    if (combine_pipeline_ == nil) {
      fail(ns_error(error, "could not create the fused MoE combine pipeline"));
    }
    gate_up_pipeline_ =
        [device_ newComputePipelineStateWithFunction:gate_up_function error:&error];
    if (gate_up_pipeline_ == nil) {
      fail(ns_error(error, "could not create the gate/up+SwiGLU pipeline"));
    }
    down_pipeline_ =
        [device_ newComputePipelineStateWithFunction:down_function error:&error];
    if (down_pipeline_ == nil) {
      fail(ns_error(error, "could not create the down/router pipeline"));
    }
    combine_router_pipeline_ =
        [device_ newComputePipelineStateWithFunction:combine_router_function
                                               error:&error];
    if (combine_router_pipeline_ == nil) {
      fail(ns_error(error, "could not create the two-stage combine pipeline"));
    }

    build_data();
    allocate_buffers();
  }

  FusedMoeMeasurement measure(
      std::size_t threadgroups,
      std::size_t threads_per_threadgroup,
      std::uint32_t warmups,
      std::uint32_t trials,
      bool two_stage) const {
    if (threadgroups == 0 || threads_per_threadgroup == 0 ||
        threads_per_threadgroup % 32 != 0) {
      fail("fused MoE requires 32-wide threadgroups");
    }
    if (threads_per_threadgroup > layer_pipeline_.maxTotalThreadsPerThreadgroup) {
      fail("fused MoE threadgroup exceeds the device pipeline limit");
    }
    const std::size_t total_simdgroups =
        threadgroups * (threads_per_threadgroup / 32);
    if (total_simdgroups % kSelectedExperts != 0) {
      fail("fused MoE SIMD groups must divide across eight experts");
    }
    const std::size_t simdgroups_per_expert = total_simdgroups / kSelectedExperts;
    if (simdgroups_per_expert == 0 ||
        kIntermediateSize % simdgroups_per_expert != 0) {
      fail("fused MoE SIMD partition does not divide intermediate channels");
    }
    const std::size_t channels = kIntermediateSize / simdgroups_per_expert;
    if (channels < 4 || channels % 4 != 0) {
      fail("fused MoE channel tile must contain at least four rows");
    }

    if (two_stage && debug_stage_ != 0) {
      fail("debug stages are only supported for the legacy megakernel");
    }
    if (two_stage) {
      const std::size_t active_output_size = kSelectedExperts * kHiddenSize;
      if (active_output_size % total_simdgroups != 0 ||
          (active_output_size / total_simdgroups) % 4 != 0) {
        fail("two-stage down schedule must divide rows into four-row tiles");
      }
      const std::size_t rows_per_simdgroup = active_output_size / total_simdgroups;
      if (kHiddenSize % rows_per_simdgroup != 0) {
        fail("two-stage down tile must not cross an expert boundary");
      }
    }

    const FusedMoeParams params{
        static_cast<std::uint32_t>(bits_ == QuantizationBits::q4 ? 4 : 8),
        static_cast<std::uint32_t>(threadgroups),
        static_cast<std::uint32_t>(kSelectedExperts),
        static_cast<std::uint32_t>(simdgroups_per_expert),
        static_cast<std::uint32_t>(channels),
        static_cast<std::uint32_t>(kHiddenSize),
        static_cast<std::uint32_t>(kIntermediateSize),
        debug_stage_,
    };
    std::memcpy(params_buffer_.contents, &params, sizeof(params));

    const auto run_once = [&]() {
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      if (two_stage) {
        [encoder setComputePipelineState:gate_up_pipeline_];
        [encoder setBuffer:gate_up_weights_buffer_ offset:0 atIndex:0];
        [encoder setBuffer:gate_up_scales_buffer_ offset:0 atIndex:1];
        [encoder setBuffer:gate_up_biases_buffer_ offset:0 atIndex:2];
        [encoder setBuffer:x_buffer_ offset:0 atIndex:3];
        [encoder setBuffer:indices_buffer_ offset:0 atIndex:4];
        [encoder setBuffer:intermediate_buffer_ offset:0 atIndex:5];
        [encoder setBuffer:params_buffer_ offset:0 atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];

        [encoder setComputePipelineState:down_pipeline_];
        [encoder setBuffer:down_weights_buffer_ offset:0 atIndex:0];
        [encoder setBuffer:down_scales_buffer_ offset:0 atIndex:1];
        [encoder setBuffer:down_biases_buffer_ offset:0 atIndex:2];
        [encoder setBuffer:intermediate_buffer_ offset:0 atIndex:3];
        [encoder setBuffer:indices_buffer_ offset:0 atIndex:4];
        [encoder setBuffer:router_buffer_ offset:0 atIndex:5];
        [encoder setBuffer:partial_buffer_ offset:0 atIndex:6];
        [encoder setBuffer:params_buffer_ offset:0 atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];

        [encoder setComputePipelineState:combine_router_pipeline_];
        [encoder setBuffer:partial_buffer_ offset:0 atIndex:0];
        [encoder setBuffer:router_buffer_ offset:0 atIndex:1];
        [encoder setBuffer:output_buffer_ offset:0 atIndex:2];
        [encoder setBuffer:params_buffer_ offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(kHiddenSize / 256, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      } else {
        [encoder setComputePipelineState:layer_pipeline_];
        [encoder setBuffer:gate_up_weights_buffer_ offset:0 atIndex:0];
        [encoder setBuffer:gate_up_scales_buffer_ offset:0 atIndex:1];
        [encoder setBuffer:gate_up_biases_buffer_ offset:0 atIndex:2];
        [encoder setBuffer:down_weights_buffer_ offset:0 atIndex:3];
        [encoder setBuffer:down_scales_buffer_ offset:0 atIndex:4];
        [encoder setBuffer:down_biases_buffer_ offset:0 atIndex:5];
        [encoder setBuffer:x_buffer_ offset:0 atIndex:6];
        [encoder setBuffer:indices_buffer_ offset:0 atIndex:7];
        [encoder setBuffer:partial_buffer_ offset:0 atIndex:8];
        [encoder setBuffer:params_buffer_ offset:0 atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];

        if (debug_stage_ == 0) {
          [encoder setComputePipelineState:combine_pipeline_];
          [encoder setBuffer:partial_buffer_ offset:0 atIndex:0];
          [encoder setBuffer:router_buffer_ offset:0 atIndex:1];
          [encoder setBuffer:output_buffer_ offset:0 atIndex:2];
          [encoder setBuffer:params_buffer_ offset:0 atIndex:3];
          [encoder dispatchThreadgroups:MTLSizeMake(kHiddenSize / 256, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
      }
      [encoder endEncoding];

      [command_buffer commit];
      const auto started = Clock::now();
      [command_buffer waitUntilCompleted];
      const auto ended = Clock::now();
      if (command_buffer.error != nil) {
        fail(ns_error(command_buffer.error, "fused MoE kernels failed"));
      }
      if (command_buffer.GPUStartTime == 0 || command_buffer.GPUEndTime == 0) {
        fail("Metal did not provide GPU timing for fused MoE");
      }
      return std::pair<double, double>{
          std::chrono::duration<double, std::micro>(ended - started).count(),
          (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1e6,
      };
    };

    for (std::uint32_t index = 0; index < warmups; ++index) {
      static_cast<void>(run_once());
    }

    FusedMoeMeasurement measurement;
    measurement.selected_experts = selected_experts_.size();
    measurement.total_experts = total_experts_;
    measurement.two_stage = two_stage;
    measurement.threadgroups = threadgroups;
    measurement.threads_per_threadgroup = threads_per_threadgroup;
    const std::size_t packed_bytes_per_gate_up_expert =
        kGateUpRowsPerExpert * kHiddenSize *
        (bits_ == QuantizationBits::q4 ? 4 : 8) / 8;
    const std::size_t packed_bytes_per_down_expert =
        kDownRowsPerExpert * kIntermediateSize *
        (bits_ == QuantizationBits::q4 ? 4 : 8) / 8;
    measurement.weight_bytes =
        selected_experts_.size() *
        (packed_bytes_per_gate_up_expert + packed_bytes_per_down_expert);
    measurement.wall_micros.reserve(trials);
    measurement.gpu_micros.reserve(trials);
    for (std::uint32_t index = 0; index < trials; ++index) {
      auto [wall, gpu] = run_once();
      measurement.wall_micros.push_back(wall);
      measurement.gpu_micros.push_back(gpu);
    }

    if (debug_stage_ == 1) {
      const auto* partial = static_cast<const float*>(partial_buffer_.contents);
      for (std::size_t slot = 0; slot < kSelectedExperts; ++slot) {
        const std::size_t expert = selected_experts_.at(slot);
        for (std::size_t channel = 0; channel < kIntermediateSize; ++channel) {
          const double actual_value = partial[expert * kIntermediateSize + channel];
          const double reference_value =
              debug_intermediate_.at(slot * kIntermediateSize + channel);
          measurement.debug_checksum += actual_value;
          measurement.debug_reference_checksum += reference_value;
          measurement.debug_max_error = std::max(
              measurement.debug_max_error, std::abs(actual_value - reference_value));
        }
      }
      return measurement;
    }

    if (debug_stage_ == 2) {
      const auto* partial = static_cast<const float*>(partial_buffer_.contents);
      for (std::size_t slot = 0; slot < kSelectedExperts; ++slot) {
        const std::size_t expert = selected_experts_.at(slot);
        for (std::size_t shard = 0; shard < simdgroups_per_expert; ++shard) {
          const std::size_t channel_start = shard * channels;
          std::vector<float> channel_values(kIntermediateSize, 0.0f);
          std::copy(
              debug_intermediate_.begin() + slot * kIntermediateSize + channel_start,
              debug_intermediate_.begin() + slot * kIntermediateSize + channel_start + channels,
              channel_values.begin() + channel_start);
          for (std::size_t row = 0; row < kHiddenSize; ++row) {
            const double actual_value =
                partial[(expert * simdgroups_per_expert + shard) * kHiddenSize + row];
            const double reference_value = row_dot(
                down_weights_, down_scales_, down_biases_,
                expert * kDownRowsPerExpert + row,
                kIntermediateSize, channel_values);
            measurement.debug_checksum += actual_value;
            measurement.debug_reference_checksum += reference_value;
            measurement.debug_max_error = std::max(
                measurement.debug_max_error, std::abs(actual_value - reference_value));
          }
        }
      }
      return measurement;
    }

    const auto* actual = static_cast<const float*>(output_buffer_.contents);
    const auto& reference = two_stage ? expected_direct_ : expected_;
    double squared_error = 0.0;
    for (std::size_t index = 0; index < kHiddenSize; ++index) {
      if (!std::isfinite(actual[index])) {
        fail("fused MoE produced a non-finite output");
      }
      measurement.output_checksum += actual[index];
      measurement.reference_checksum += reference[index];
      const double error =
          static_cast<double>(actual[index]) - reference[index];
      measurement.max_absolute_error =
          std::max(measurement.max_absolute_error, std::abs(error));
      squared_error += error * error;
    }
    measurement.rmse = std::sqrt(squared_error / kHiddenSize);
    if (!two_stage) {
      const auto* partial = static_cast<const float*>(partial_buffer_.contents);
      for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
        double combined = 0.0;
        for (std::size_t expert = 0; expert < kSelectedExperts; ++expert) {
          double expert_sum = 0.0;
          for (std::size_t shard = 0; shard < simdgroups_per_expert; ++shard) {
            expert_sum += partial[
                (expert * simdgroups_per_expert + shard) * kHiddenSize + hidden];
          }
          combined += router_weights_.at(expert) * expert_sum;
        }
        measurement.debug_checksum += combined;
        measurement.debug_reference_checksum += expected_.at(hidden);
        measurement.debug_max_error = std::max(
            measurement.debug_max_error,
            std::abs(combined - static_cast<double>(actual[hidden])));
      }
    }
    return measurement;
  }

 private:
  void build_data() {
    if (!dataset_path_.empty()) {
      load_dataset();
      return;
    }
    Random random(bits_ == QuantizationBits::q4
                      ? 0xF05EEDB1ULL
                      : 0xF05EEDB2ULL);
    x_.resize(kHiddenSize);
    x_values_.resize(kHiddenSize);
    for (std::size_t index = 0; index < kHiddenSize; ++index) {
      const float value = random.uniform(-1.0f, 1.0f);
      x_.at(index) = float_to_bf16(value);
      x_values_.at(index) = bf16_to_float(x_.at(index));
    }

    const std::size_t gate_up_rows = total_experts_ * kGateUpRowsPerExpert;
    const std::size_t down_rows = total_experts_ * kDownRowsPerExpert;
    const std::size_t gate_up_stride = kHiddenSize / values_per_pack_;
    const std::size_t down_stride = kIntermediateSize / values_per_pack_;
    gate_up_weights_.resize(gate_up_rows * gate_up_stride);
    down_weights_.resize(down_rows * down_stride);
    gate_up_scales_.resize(gate_up_rows * (kHiddenSize / kQuantGroupSize));
    gate_up_biases_.resize(gate_up_scales_.size());
    down_scales_.resize(down_rows * (kIntermediateSize / kQuantGroupSize));
    down_biases_.resize(down_scales_.size());

    fill_bank(
        gate_up_weights_, gate_up_scales_, gate_up_biases_, kHiddenSize, random);
    fill_bank(
        down_weights_, down_scales_, down_biases_, kIntermediateSize, random);

    router_weights_.resize(kSelectedExperts);
    float router_sum = 0.0f;
    for (float& weight : router_weights_) {
      weight = random.uniform(0.5f, 1.5f);
      router_sum += weight;
    }
    for (float& weight : router_weights_) {
      weight /= router_sum;
    }

    compute_references();
  }

  void compute_references() {
    expected_.assign(kHiddenSize, 0.0f);
    expected_direct_.assign(kHiddenSize, 0.0f);
    debug_intermediate_.assign(kSelectedExperts * kIntermediateSize, 0.0f);
    for (std::size_t slot = 0; slot < kSelectedExperts; ++slot) {
      const std::size_t expert = selected_experts_.at(slot);
      std::vector<float> intermediate(kIntermediateSize);
      for (std::size_t channel = 0; channel < kIntermediateSize; ++channel) {
        const std::size_t up_row = expert * kGateUpRowsPerExpert + channel;
        const std::size_t gate_row = up_row + kIntermediateSize;
        const float up = row_dot(
            gate_up_weights_, gate_up_scales_, gate_up_biases_, up_row,
            kHiddenSize);
        const float gate = row_dot(
            gate_up_weights_, gate_up_scales_, gate_up_biases_, gate_row,
            kHiddenSize);
        intermediate.at(channel) =
            (gate / (1.0f + std::exp(-gate))) * up;
        debug_intermediate_.at(slot * kIntermediateSize + channel) =
            intermediate.at(channel);
      }
      for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
        const float direct_value = row_dot(
            down_weights_, down_scales_, down_biases_,
            expert * kDownRowsPerExpert + hidden, kIntermediateSize,
            intermediate);
        expected_direct_.at(hidden) += router_weights_.at(slot) * direct_value;
        float expert_sum = 0.0f;
        for (std::size_t shard = 0; shard < 32; ++shard) {
          const std::size_t channel_start = shard * 16;
          std::vector<float> channel_values(kIntermediateSize, 0.0f);
          std::copy(
              intermediate.begin() + channel_start,
              intermediate.begin() + channel_start + 16,
              channel_values.begin() + channel_start);
          expert_sum += row_dot(
              down_weights_, down_scales_, down_biases_,
              expert * kDownRowsPerExpert + hidden, kIntermediateSize,
              channel_values);
        }
        expected_.at(hidden) += router_weights_.at(slot) * expert_sum;
      }
    }
  }

  void load_dataset() {
    std::ifstream stream(dataset_path_, std::ios::binary);
    if (!stream) {
      fail("could not open fused MoE dataset: " + dataset_path_);
    }
    FusedDatasetHeader header;
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!stream || std::memcmp(header.magic, "RNLMOE01", 8) != 0 ||
        header.version != 1 || header.bits != (bits_ == QuantizationBits::q4 ? 4 : 8) ||
        header.total_experts != total_experts_ ||
        header.selected_expert_count != kSelectedExperts ||
        header.gate_up_rows_per_expert != kGateUpRowsPerExpert ||
        header.down_rows_per_expert != kDownRowsPerExpert ||
        header.hidden_size != kHiddenSize || header.intermediate_size != kIntermediateSize) {
      fail("fused MoE dataset header does not match the requested probe");
    }

    std::vector<std::uint32_t> selected(kSelectedExperts);
    read_vector(stream, selected);
    if (selected != selected_experts_) {
      fail("fused MoE dataset expert selection does not match the probe");
    }

    const std::size_t gate_up_packed_stride = kHiddenSize / values_per_pack_;
    const std::size_t down_packed_stride = kIntermediateSize / values_per_pack_;
    const std::size_t gate_up_groups = kHiddenSize / kQuantGroupSize;
    const std::size_t down_groups = kIntermediateSize / kQuantGroupSize;
    gate_up_weights_.resize(total_experts_ * kGateUpRowsPerExpert * gate_up_packed_stride);
    gate_up_scales_.resize(total_experts_ * kGateUpRowsPerExpert * gate_up_groups);
    gate_up_biases_.resize(gate_up_scales_.size());
    down_weights_.resize(total_experts_ * kDownRowsPerExpert * down_packed_stride);
    down_scales_.resize(total_experts_ * kDownRowsPerExpert * down_groups);
    down_biases_.resize(down_scales_.size());
    x_.resize(kHiddenSize);
    x_values_.resize(kHiddenSize);
    router_weights_.resize(kSelectedExperts);

    read_vector(stream, gate_up_weights_);
    read_vector(stream, gate_up_scales_);
    read_vector(stream, gate_up_biases_);
    read_vector(stream, down_weights_);
    read_vector(stream, down_scales_);
    read_vector(stream, down_biases_);
    read_vector(stream, x_);
    read_vector(stream, router_weights_);
    for (std::size_t index = 0; index < x_.size(); ++index) {
      x_values_.at(index) = bf16_to_float(x_.at(index));
    }
    if (!stream) {
      fail("fused MoE dataset is truncated: " + dataset_path_);
    }
    compute_references();
  }

  template <typename T>
  static void read_vector(std::ifstream& stream, std::vector<T>& values) {
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
  }

  void fill_bank(
      std::vector<std::uint32_t>& weights,
      std::vector<std::uint16_t>& scales,
      std::vector<std::uint16_t>& biases,
      std::size_t reduction_size,
      Random& random) const {
    const std::size_t packed_stride = reduction_size / values_per_pack_;
    const std::size_t rows = weights.size() / packed_stride;
    const std::size_t quant_groups = reduction_size / kQuantGroupSize;
    const float scale_low = bits_ == QuantizationBits::q4 ? 0.002f : 0.00005f;
    const float scale_high = bits_ == QuantizationBits::q4 ? 0.008f : 0.00015f;
    const std::uint32_t maximum =
        bits_ == QuantizationBits::q4 ? 0x0FU : 0xFFU;
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t group = 0; group < quant_groups; ++group) {
        scales.at(row * quant_groups + group) = float_to_bf16(
            random.uniform(scale_low, scale_high));
        biases.at(row * quant_groups + group) =
            float_to_bf16(random.uniform(-0.0005f, 0.0005f));
      }
      for (std::size_t packed = 0; packed < packed_stride; ++packed) {
        std::uint32_t value = 0;
        for (std::size_t byte = 0; byte < 4; ++byte) {
          value |= (random.next_u32() & maximum) << (byte * 8);
        }
        weights.at(row * packed_stride + packed) = value;
      }
    }
  }

  float row_dot(
      const std::vector<std::uint32_t>& weights,
      const std::vector<std::uint16_t>& scales,
      const std::vector<std::uint16_t>& biases,
      std::size_t row,
      std::size_t reduction_size) const {
    const std::size_t packed_stride = reduction_size / values_per_pack_;
    const std::size_t quant_groups = reduction_size / kQuantGroupSize;
    const std::uint32_t maximum =
        bits_ == QuantizationBits::q4 ? 0x0FU : 0xFFU;
    float result = 0.0f;
    for (std::size_t group = 0; group < quant_groups; ++group) {
      const float scale = bf16_to_float(scales.at(row * quant_groups + group));
      const float bias = bf16_to_float(biases.at(row * quant_groups + group));
      float weighted = 0.0f;
      float x_sum = 0.0f;
      const std::size_t start = group * kQuantGroupSize;
      for (std::size_t index = 0; index < kQuantGroupSize; ++index) {
        const std::size_t reduction = start + index;
        const std::size_t packed = row * packed_stride + reduction / values_per_pack_;
        const std::size_t shift = bits_ == QuantizationBits::q4
            ? 4 * (reduction % values_per_pack_)
            : 8 * (reduction % values_per_pack_);
        const std::uint32_t quantized =
            (weights.at(packed) >> shift) & maximum;
        weighted += static_cast<float>(quantized) * x_values_.at(reduction);
        x_sum += x_values_.at(reduction);
      }
      result += weighted * scale + x_sum * bias;
    }
    return result;
  }

  float row_dot(
      const std::vector<std::uint32_t>& weights,
      const std::vector<std::uint16_t>& scales,
      const std::vector<std::uint16_t>& biases,
      std::size_t row,
      std::size_t reduction_size,
      const std::vector<float>& reduction_values) const {
    const std::size_t packed_stride = reduction_size / values_per_pack_;
    const std::size_t quant_groups = reduction_size / kQuantGroupSize;
    const std::uint32_t maximum =
        bits_ == QuantizationBits::q4 ? 0x0FU : 0xFFU;
    float result = 0.0f;
    for (std::size_t group = 0; group < quant_groups; ++group) {
      const float scale = bf16_to_float(scales.at(row * quant_groups + group));
      const float bias = bf16_to_float(biases.at(row * quant_groups + group));
      float weighted = 0.0f;
      float input_sum = 0.0f;
      const std::size_t start = group * kQuantGroupSize;
      for (std::size_t index = 0; index < kQuantGroupSize; ++index) {
        const std::size_t reduction = start + index;
        const std::size_t packed = row * packed_stride + reduction / values_per_pack_;
        const std::size_t shift = bits_ == QuantizationBits::q4
            ? 4 * (reduction % values_per_pack_)
            : 8 * (reduction % values_per_pack_);
        const std::uint32_t quantized =
            (weights.at(packed) >> shift) & maximum;
        weighted += static_cast<float>(quantized) * reduction_values.at(reduction);
        input_sum += reduction_values.at(reduction);
      }
      result += weighted * scale + input_sum * bias;
    }
    return result;
  }

  void allocate_buffers() {
    gate_up_weights_buffer_ = [device_
        newBufferWithBytes:gate_up_weights_.data()
                     length:gate_up_weights_.size() * sizeof(std::uint32_t)
                    options:MTLResourceStorageModeShared];
    gate_up_scales_buffer_ = [device_
        newBufferWithBytes:gate_up_scales_.data()
                     length:gate_up_scales_.size() * sizeof(std::uint16_t)
                    options:MTLResourceStorageModeShared];
    gate_up_biases_buffer_ = [device_
        newBufferWithBytes:gate_up_biases_.data()
                     length:gate_up_biases_.size() * sizeof(std::uint16_t)
                    options:MTLResourceStorageModeShared];
    down_weights_buffer_ = [device_
        newBufferWithBytes:down_weights_.data()
                     length:down_weights_.size() * sizeof(std::uint32_t)
                    options:MTLResourceStorageModeShared];
    down_scales_buffer_ = [device_
        newBufferWithBytes:down_scales_.data()
                     length:down_scales_.size() * sizeof(std::uint16_t)
                    options:MTLResourceStorageModeShared];
    down_biases_buffer_ = [device_
        newBufferWithBytes:down_biases_.data()
                     length:down_biases_.size() * sizeof(std::uint16_t)
                    options:MTLResourceStorageModeShared];
    x_buffer_ = [device_ newBufferWithBytes:x_.data()
                                      length:x_.size() * sizeof(std::uint16_t)
                                     options:MTLResourceStorageModeShared];
    indices_buffer_ = [device_
        newBufferWithBytes:selected_experts_.data()
                     length:selected_experts_.size() * sizeof(std::uint32_t)
                    options:MTLResourceStorageModeShared];
    router_buffer_ = [device_
        newBufferWithBytes:router_weights_.data()
                     length:router_weights_.size() * sizeof(float)
                    options:MTLResourceStorageModeShared];
    intermediate_buffer_ = [device_
        newBufferWithLength:kSelectedExperts * kIntermediateSize * sizeof(float)
                    options:MTLResourceStorageModeShared];
    partial_buffer_ = [device_
        newBufferWithLength:kSelectedExperts * kHiddenSize * 64 * sizeof(float)
                    options:MTLResourceStorageModeShared];
    output_buffer_ = [device_ newBufferWithLength:kHiddenSize * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    params_buffer_ = [device_ newBufferWithLength:sizeof(FusedMoeParams)
                                          options:MTLResourceStorageModeShared];
    if (gate_up_weights_buffer_ == nil || gate_up_scales_buffer_ == nil ||
        gate_up_biases_buffer_ == nil || down_weights_buffer_ == nil ||
        down_scales_buffer_ == nil || down_biases_buffer_ == nil ||
        x_buffer_ == nil || indices_buffer_ == nil || router_buffer_ == nil ||
        intermediate_buffer_ == nil || partial_buffer_ == nil || output_buffer_ == nil ||
        params_buffer_ == nil) {
      fail("could not allocate fused MoE buffers");
    }
  }

  QuantizationBits bits_;
  std::size_t total_experts_;
  std::vector<std::uint32_t> selected_experts_;
  std::size_t values_per_pack_;
  std::vector<std::uint32_t> gate_up_weights_;
  std::vector<std::uint16_t> gate_up_scales_;
  std::vector<std::uint16_t> gate_up_biases_;
  std::vector<std::uint32_t> down_weights_;
  std::vector<std::uint16_t> down_scales_;
  std::vector<std::uint16_t> down_biases_;
  std::vector<std::uint16_t> x_;
  std::vector<float> x_values_;
  std::vector<float> router_weights_;
  std::vector<float> expected_;
  std::vector<float> expected_direct_;
  std::vector<float> debug_intermediate_;
  std::uint32_t debug_stage_{};
  std::string dataset_path_;
  __strong id<MTLDevice> device_{};
  __strong id<MTLCommandQueue> queue_{};
  __strong id<MTLComputePipelineState> layer_pipeline_{};
  __strong id<MTLComputePipelineState> combine_pipeline_{};
  __strong id<MTLComputePipelineState> gate_up_pipeline_{};
  __strong id<MTLComputePipelineState> down_pipeline_{};
  __strong id<MTLComputePipelineState> combine_router_pipeline_{};
  __strong id<MTLBuffer> gate_up_weights_buffer_{};
  __strong id<MTLBuffer> gate_up_scales_buffer_{};
  __strong id<MTLBuffer> gate_up_biases_buffer_{};
  __strong id<MTLBuffer> down_weights_buffer_{};
  __strong id<MTLBuffer> down_scales_buffer_{};
  __strong id<MTLBuffer> down_biases_buffer_{};
  __strong id<MTLBuffer> x_buffer_{};
  __strong id<MTLBuffer> indices_buffer_{};
  __strong id<MTLBuffer> router_buffer_{};
  __strong id<MTLBuffer> intermediate_buffer_{};
  __strong id<MTLBuffer> partial_buffer_{};
  __strong id<MTLBuffer> output_buffer_{};
  __strong id<MTLBuffer> params_buffer_{};
};

FusedMoeProbe::FusedMoeProbe(
    QuantizationBits bits,
    std::size_t total_experts,
    std::vector<std::uint32_t> selected_experts,
    std::uint32_t debug_stage,
    std::string dataset_path)
    : impl_(std::make_unique<Impl>(
          bits,
          total_experts,
          std::move(selected_experts),
          debug_stage,
          std::move(dataset_path))) {}

FusedMoeProbe::~FusedMoeProbe() = default;
FusedMoeProbe::FusedMoeProbe(FusedMoeProbe&&) noexcept = default;
FusedMoeProbe& FusedMoeProbe::operator=(FusedMoeProbe&&) noexcept = default;

FusedMoeMeasurement FusedMoeProbe::measure(
    std::size_t threadgroups,
    std::size_t threads_per_threadgroup,
    std::uint32_t warmups,
    std::uint32_t trials,
    bool two_stage) const {
  return impl_->measure(
      threadgroups, threads_per_threadgroup, warmups, trials, two_stage);
}

}  // namespace runnel::metal
