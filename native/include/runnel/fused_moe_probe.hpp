#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runnel/quantized_qmv_probe.hpp"

namespace runnel::metal {

struct FusedMoeMeasurement {
  std::size_t selected_experts{};
  std::size_t total_experts{};
  std::size_t threadgroups{};
  std::size_t threads_per_threadgroup{};
  std::size_t weight_bytes{};
  bool two_stage{};
  double output_checksum{};
  double reference_checksum{};
  double max_absolute_error{};
  double rmse{};
  double debug_checksum{};
  double debug_reference_checksum{};
  double debug_max_error{};
  std::vector<double> wall_micros;
  std::vector<double> gpu_micros;
};

class FusedMoeProbe {
 public:
  FusedMoeProbe(
      QuantizationBits bits,
      std::size_t total_experts,
      std::vector<std::uint32_t> selected_experts,
      std::uint32_t debug_stage = 0,
      std::string dataset_path = {});
  ~FusedMoeProbe();

  FusedMoeProbe(const FusedMoeProbe&) = delete;
  FusedMoeProbe& operator=(const FusedMoeProbe&) = delete;
  FusedMoeProbe(FusedMoeProbe&&) noexcept;
  FusedMoeProbe& operator=(FusedMoeProbe&&) noexcept;

  [[nodiscard]] FusedMoeMeasurement measure(
      std::size_t threadgroups,
      std::size_t threads_per_threadgroup,
      std::uint32_t warmups,
      std::uint32_t trials,
      bool two_stage = false) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runnel::metal
