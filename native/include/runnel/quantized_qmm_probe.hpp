#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "runnel/quantized_qmv_probe.hpp"

namespace runnel::metal {

struct QmmMeasurement {
  std::size_t token_count{};
  std::size_t output_count{};
  std::size_t reduction_size{};
  std::size_t token_tile{8};
  std::size_t output_tile{32};
  std::size_t reduction_tile{256};
  std::size_t threadgroups{};
  std::size_t weight_bytes{};
  double output_checksum{};
  double reference_checksum{};
  double max_absolute_error{};
  double rmse{};
  std::vector<double> wall_micros;
  std::vector<double> gpu_micros;
};

class QuantizedQmmProbe {
 public:
  QuantizedQmmProbe(
      QuantizationBits bits,
      std::size_t token_count,
      std::size_t output_count,
      std::size_t reduction_size,
      std::size_t quant_group_size = 64);
  ~QuantizedQmmProbe();

  QuantizedQmmProbe(const QuantizedQmmProbe&) = delete;
  QuantizedQmmProbe& operator=(const QuantizedQmmProbe&) = delete;
  QuantizedQmmProbe(QuantizedQmmProbe&&) noexcept;
  QuantizedQmmProbe& operator=(QuantizedQmmProbe&&) noexcept;

  [[nodiscard]] QmmMeasurement measure(
      std::uint32_t warmups,
      std::uint32_t trials) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runnel::metal
