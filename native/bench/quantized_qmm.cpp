#include "runnel/quantized_qmm_probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  runnel::metal::QuantizationBits bits =
      runnel::metal::QuantizationBits::q4;
  std::size_t token_count = 64;
  std::size_t output_count = 4096;
  std::size_t reduction_size = 2048;
  std::size_t quant_group_size = 64;
  std::uint32_t warmups = 2;
  std::uint32_t trials = 7;
};

std::string value_for(int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(argv[index]) + " requires a value");
  }
  return argv[++index];
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--bits") {
      const std::string value = value_for(argc, argv, index);
      if (value == "q4") {
        options.bits = runnel::metal::QuantizationBits::q4;
      } else if (value == "q8") {
        options.bits = runnel::metal::QuantizationBits::q8;
      } else {
        throw std::invalid_argument("--bits must be q4 or q8");
      }
    } else if (argument == "--tokens") {
      options.token_count = std::stoull(value_for(argc, argv, index));
    } else if (argument == "--outputs") {
      options.output_count = std::stoull(value_for(argc, argv, index));
    } else if (argument == "--reduction-size") {
      options.reduction_size = std::stoull(value_for(argc, argv, index));
    } else if (argument == "--quant-group-size") {
      options.quant_group_size = std::stoull(value_for(argc, argv, index));
    } else if (argument == "--warmups") {
      options.warmups = static_cast<std::uint32_t>(
          std::stoul(value_for(argc, argv, index)));
    } else if (argument == "--trials") {
      options.trials = static_cast<std::uint32_t>(
          std::stoul(value_for(argc, argv, index)));
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: runnel-metal-qmm [--bits q4|q8] [--tokens N] "
                   "[--outputs N] [--reduction-size N] "
                   "[--quant-group-size N] [--warmups N] [--trials N]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (options.trials == 0) {
    throw std::invalid_argument("--trials must be positive");
  }
  return options;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[middle - 1] + values[middle]) / 2.0;
  }
  return values[middle];
}

void print_trials(const std::vector<double>& wall, const std::vector<double>& gpu) {
  std::cout << "[";
  for (std::size_t index = 0; index < gpu.size(); ++index) {
    if (index != 0) {
      std::cout << ", ";
    }
    std::cout << "{\"wall_micros\": " << wall[index]
              << ", \"gpu_micros\": " << gpu[index] << "}";
  }
  std::cout << "]";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    runnel::metal::QuantizedQmmProbe probe(
        options.bits,
        options.token_count,
        options.output_count,
        options.reduction_size,
        options.quant_group_size);
    const auto measurement = probe.measure(options.warmups, options.trials);
    if (measurement.max_absolute_error > 0.2 || measurement.rmse > 0.05) {
      std::cerr << "qmm max_error=" << measurement.max_absolute_error
                << " rmse=" << measurement.rmse
                << " checksum=" << measurement.output_checksum
                << " reference=" << measurement.reference_checksum << '\n';
      throw std::runtime_error("quantized QMM failed CPU-reference validation");
    }
    const double gpu_median = median(measurement.gpu_micros);
    const double wall_median = median(measurement.wall_micros);
    const double gib_per_second = static_cast<double>(measurement.weight_bytes) /
        (gpu_median * 1e3);
    const char* bits = options.bits == runnel::metal::QuantizationBits::q4
                           ? "q4"
                           : "q8";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "{\n  \"protocol\": {\"bits\": \"" << bits
              << "\", \"tokens\": " << options.token_count
              << ", \"outputs\": " << options.output_count
              << ", \"reduction_size\": " << options.reduction_size
              << ", \"token_tile\": 8, \"output_tile\": 32"
              << ", \"reduction_tile\": 256"
              << ", \"warmups\": " << options.warmups
              << ", \"trials\": " << options.trials
              << "},\n  \"measurement\": {\"threadgroups\": "
              << measurement.threadgroups
              << ", \"weight_bytes\": " << measurement.weight_bytes
              << ", \"weight_gib_per_second\": " << gib_per_second
              << ", \"median_wall_micros\": " << wall_median
              << ", \"median_gpu_micros\": " << gpu_median
              << ", \"output_checksum\": " << measurement.output_checksum
              << ", \"reference_checksum\": " << measurement.reference_checksum
              << ", \"max_absolute_error\": " << measurement.max_absolute_error
              << ", \"rmse\": " << measurement.rmse << ", \"trials\": ";
    print_trials(measurement.wall_micros, measurement.gpu_micros);
    std::cout << "}\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "runnel-metal-qmm: " << error.what() << '\n';
    return 1;
  }
}
