#include "runnel/metal_probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::size_t buffer_bytes = std::size_t{4} << 30;
  std::uint32_t warmups = 2;
  std::uint32_t trials = 7;
  bool quick = false;
};

std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        if (character < 0x20) {
          static constexpr char hex[] = "0123456789abcdef";
          escaped += "\\u00";
          escaped.push_back(hex[character >> 4]);
          escaped.push_back(hex[character & 0x0F]);
        } else {
          escaped.push_back(static_cast<char>(character));
        }
    }
  }
  return escaped;
}

std::uint64_t parse_u64(const std::string& text, const std::string& name) {
  std::size_t consumed = 0;
  const auto value = std::stoull(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(name + " must be an integer");
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&]() -> std::string {
      if (index + 1 >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      return argv[++index];
    };

    if (argument == "--bytes-gib") {
      const double gib = std::stod(value());
      if (!(gib > 0.0) || !std::isfinite(gib)) {
        throw std::invalid_argument("--bytes-gib must be positive");
      }
      const auto bytes = static_cast<std::uint64_t>(gib * (1ULL << 30));
      options.buffer_bytes = static_cast<std::size_t>(bytes & ~std::uint64_t{4095});
    } else if (argument == "--warmups") {
      options.warmups = static_cast<std::uint32_t>(
          parse_u64(value(), "--warmups"));
    } else if (argument == "--trials") {
      options.trials = static_cast<std::uint32_t>(
          parse_u64(value(), "--trials"));
    } else if (argument == "--quick") {
      options.quick = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "Usage: runnel-metal-stream [--bytes-gib N] [--warmups N] "
             "[--trials N] [--quick]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }

  if (options.buffer_bytes == 0 || options.buffer_bytes % 4096 != 0) {
    throw std::invalid_argument(
        "buffer size must be a non-zero multiple of 4096");
  }
  if (options.trials == 0) {
    throw std::invalid_argument("--trials must be positive");
  }
  return options;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize an empty sample");
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[middle - 1] + values[middle]) / 2.0;
  }
  return values[middle];
}

std::vector<std::pair<std::size_t, std::size_t>> configurations(
    bool quick) {
  std::set<std::pair<std::size_t, std::size_t>> unique;
  if (quick) {
    for (const std::size_t threads : {std::size_t{256}, std::size_t{1024}}) {
      for (const std::size_t groups :
           {std::size_t{1}, std::size_t{8}, std::size_t{32}}) {
        unique.emplace(groups, threads);
      }
    }
  } else {
    for (const std::size_t threads :
         {std::size_t{64}, std::size_t{128}, std::size_t{256},
          std::size_t{512}, std::size_t{1024}}) {
      unique.emplace(1, threads);
    }
    for (const std::size_t groups :
         {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8},
          std::size_t{16}, std::size_t{32}}) {
      for (const std::size_t threads :
           {std::size_t{256}, std::size_t{1024}}) {
        unique.emplace(groups, threads);
      }
    }
  }
  return {unique.begin(), unique.end()};
}

void print_measurement(const runnel::metal::StreamMeasurement& measurement) {
  std::vector<double> wall;
  std::vector<double> gpu;
  wall.reserve(measurement.trials.size());
  gpu.reserve(measurement.trials.size());
  for (const auto& trial : measurement.trials) {
    wall.push_back(trial.wall_micros);
    gpu.push_back(trial.gpu_micros);
  }
  const double wall_median = median(wall);
  const double gpu_median = median(gpu);
  const double gib_per_second =
      static_cast<double>(measurement.bytes) / (gpu_median * 1e3);

  std::cout << "    {\"threadgroups\": " << measurement.threadgroups
            << ", \"threads_per_threadgroup\": "
            << measurement.threads_per_threadgroup
            << ", \"checksum\": " << measurement.checksum
            << ", \"bytes_per_second_gib\": " << gib_per_second
            << ", \"median_wall_micros\": " << wall_median
            << ", \"median_gpu_micros\": " << gpu_median << ", \"trials\": [";
  for (std::size_t index = 0; index < measurement.trials.size(); ++index) {
    if (index != 0) {
      std::cout << ", ";
    }
    std::cout << "{\"wall_micros\": " << measurement.trials[index].wall_micros
              << ", \"gpu_micros\": "
              << measurement.trials[index].gpu_micros << "}";
  }
  std::cout << "]}";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    runnel::metal::StreamProbe probe(options.buffer_bytes);
    const auto device = probe.device_info();
    const auto configs = configurations(options.quick);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "{\n  \"device\": {\"name\": \""
              << json_escape(device.name) << "\", \"core_count\": "
              << device.core_count << ", \"max_threads_per_threadgroup\": "
              << device.max_threads_per_threadgroup
              << ", \"threads_per_warp\": " << device.max_threads_per_warp
              << ", \"recommended_working_set_bytes\": "
              << device.recommended_working_set_bytes
              << ", \"low_power\": " << (device.low_power ? "true" : "false")
              << ", \"headless\": " << (device.headless ? "true" : "false")
              << ", \"unified_memory\": "
              << (device.has_unified_memory ? "true" : "false")
              << "},\n  \"protocol\": {\"buffer_bytes\": " << options.buffer_bytes
              << ", \"warmups\": " << options.warmups
              << ", \"trials\": " << options.trials
              << ", \"quick\": " << (options.quick ? "true" : "false")
              << "},\n  \"measurements\": [\n";

    std::uint32_t expected_checksum = 0;
    bool have_checksum = false;
    bool checksum_mismatch = false;
    for (std::size_t index = 0; index < configs.size(); ++index) {
      const auto [groups, threads] = configs[index];
      const auto measurement = probe.measure(
          groups, threads, options.warmups, options.trials);
      if (!have_checksum) {
        expected_checksum = measurement.checksum;
        have_checksum = true;
      } else if (measurement.checksum != expected_checksum) {
        checksum_mismatch = true;
      }
      print_measurement(measurement);
      std::cout << (index + 1 == configs.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
    if (checksum_mismatch) {
      throw std::runtime_error("stream checksum mismatch across launch shapes");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "runnel-metal-stream: " << error.what() << '\n';
    return 1;
  }
}
