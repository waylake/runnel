#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace runnel::metal {

enum class QuantizationBits {
  q4,
  q8,
};

struct QmvMeasurement {
  std::size_t output_size{};
  std::size_t reduction_size{};
  std::size_t threadgroups{};
  std::size_t threads_per_threadgroup{};
  std::size_t weight_bytes{};
  double output_checksum{};
  double reference_checksum{};
  double max_reference_magnitude{};
  double max_absolute_error{};
  double max_relative_error{};
  double rmse{};
  std::vector<float> first_actual;
  std::vector<float> first_reference;
  std::vector<double> wall_micros;
  std::vector<double> gpu_micros;
};

class QuantizedQmvProbe {
 public:
  QuantizedQmvProbe(
      QuantizationBits bits,
      std::size_t output_size,
      std::size_t reduction_size,
      std::size_t quant_group_size,
      std::size_t weight_bank_output_size = 0,
      std::size_t rows_per_expert = 0,
      std::vector<std::uint32_t> selected_experts = {});
  ~QuantizedQmvProbe();

  QuantizedQmvProbe(const QuantizedQmvProbe&) = delete;
  QuantizedQmvProbe& operator=(const QuantizedQmvProbe&) = delete;
  QuantizedQmvProbe(QuantizedQmvProbe&&) noexcept;
  QuantizedQmvProbe& operator=(QuantizedQmvProbe&&) noexcept;

  [[nodiscard]] QmvMeasurement measure(
      std::size_t threadgroups,
      std::size_t threads_per_threadgroup,
      std::uint32_t warmups,
      std::uint32_t trials) const;
  void write_dataset(const std::string& path) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runnel::metal
