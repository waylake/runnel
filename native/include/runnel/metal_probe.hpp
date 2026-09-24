#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace runnel::metal {

struct DeviceInfo {
  std::string name;
  std::uint32_t core_count{};
  std::size_t max_threads_per_threadgroup{};
  std::size_t max_threads_per_warp{};
  std::uint64_t recommended_working_set_bytes{};
  bool low_power{};
  bool headless{};
  bool has_unified_memory{};
};

struct StreamTrial {
  double wall_micros{};
  double gpu_micros{};
};

struct StreamMeasurement {
  std::size_t bytes{};
  std::size_t threadgroups{};
  std::size_t threads_per_threadgroup{};
  std::uint32_t checksum{};
  std::vector<StreamTrial> trials;
};

class StreamProbe {
 public:
  explicit StreamProbe(std::size_t buffer_bytes);
  ~StreamProbe();

  StreamProbe(const StreamProbe&) = delete;
  StreamProbe& operator=(const StreamProbe&) = delete;
  StreamProbe(StreamProbe&&) noexcept;
  StreamProbe& operator=(StreamProbe&&) noexcept;

  [[nodiscard]] DeviceInfo device_info() const;
  [[nodiscard]] StreamMeasurement measure(
      std::size_t threadgroups,
      std::size_t threads_per_threadgroup,
      std::uint32_t warmups,
      std::uint32_t trials) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace runnel::metal
