#include "runnel/fused_moe_probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  runnel::metal::QuantizationBits bits =
      runnel::metal::QuantizationBits::q4;
  std::size_t total_experts = 64;
  std::vector<std::uint32_t> selected_experts{3, 17, 29, 42, 50, 61, 7, 55};
  std::size_t threadgroups = 32;
  std::vector<std::size_t> thread_sweep{256, 512};
  std::uint32_t warmups = 5;
  std::uint32_t trials = 20;
  std::uint32_t debug_stage = 0;
  bool two_stage = false;
  std::string dataset_path;
};

std::string value_for(int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(argv[index]) + " requires a value");
  }
  return argv[++index];
}

std::vector<std::uint32_t> parse_experts(const std::string& text) {
  std::vector<std::uint32_t> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::string token = text.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    if (token.empty()) {
      throw std::invalid_argument("--experts contains an empty value");
    }
    const unsigned long parsed = std::stoul(token);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("--experts value exceeds uint32");
    }
    result.push_back(static_cast<std::uint32_t>(parsed));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return result;
}

std::vector<std::size_t> parse_threads(const std::string& text) {
  std::vector<std::size_t> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::string token = text.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    if (token.empty()) {
      throw std::invalid_argument("--threads contains an empty value");
    }
    result.push_back(std::stoul(token));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return result;
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
    } else if (argument == "--total-experts") {
      options.total_experts = std::stoull(value_for(argc, argv, index));
    } else if (argument == "--experts") {
      options.selected_experts = parse_experts(value_for(argc, argv, index));
    } else if (argument == "--groups") {
      options.threadgroups = std::stoull(value_for(argc, argv, index));
    } else if (argument == "--threads") {
      options.thread_sweep = parse_threads(value_for(argc, argv, index));
    } else if (argument == "--warmups") {
      options.warmups = static_cast<std::uint32_t>(
          std::stoul(value_for(argc, argv, index)));
    } else if (argument == "--trials") {
      options.trials = static_cast<std::uint32_t>(
          std::stoul(value_for(argc, argv, index)));
    } else if (argument == "--debug-stage") {
      options.debug_stage = static_cast<std::uint32_t>(
          std::stoul(value_for(argc, argv, index)));
    } else if (argument == "--two-stage") {
      options.two_stage = true;
    } else if (argument == "--dataset") {
      options.dataset_path = value_for(argc, argv, index);
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: runnel-metal-fused-moe [--bits q4|q8] "
                   "[--total-experts N] [--experts I,J,...] [--groups N] "
                   "[--threads 256,512] [--two-stage] [--dataset PATH] "
                   "[--warmups N] [--trials N]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (options.total_experts == 0 || options.selected_experts.size() != 8 ||
      options.trials == 0) {
    throw std::invalid_argument(
        "fused MoE requires experts, eight selections, and positive trials");
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
    runnel::metal::FusedMoeProbe probe(
        options.bits,
        options.total_experts,
        options.selected_experts,
        options.debug_stage,
        options.dataset_path);
    const char* bits = options.bits == runnel::metal::QuantizationBits::q4
                           ? "q4"
                           : "q8";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "{\n  \"protocol\": {\"bits\": \"" << bits
              << "\", \"total_experts\": " << options.total_experts
              << ", \"selected_experts\": [";
    for (std::size_t index = 0; index < options.selected_experts.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << options.selected_experts[index];
    }
    std::cout << "], \"groups\": " << options.threadgroups
              << ", \"schedule\": \""
              << (options.two_stage ? "two-stage" : "megakernel")
              << "\", \"dataset_loaded\": "
              << (options.dataset_path.empty() ? "false" : "true")
              << ", \"warmups\": " << options.warmups
              << ", \"trials\": " << options.trials
              << "},  \"measurements\": [\n";

    for (std::size_t index = 0; index < options.thread_sweep.size(); ++index) {
      const auto measurement = probe.measure(
          options.threadgroups,
          options.thread_sweep[index],
          options.warmups,
          options.trials,
          options.two_stage);
      const double max_error_limit = options.two_stage ? 1.0 : 0.2;
      const double rmse_limit = options.two_stage ? 0.2 : 0.05;
      const double checksum_limit = options.two_stage
          ? std::max(5.0, std::abs(measurement.reference_checksum) * 0.25)
          : 1.0;
      if (measurement.max_absolute_error > max_error_limit ||
          measurement.rmse > rmse_limit ||
          std::abs(measurement.output_checksum - measurement.reference_checksum) >
              checksum_limit) {
        std::cerr << "debug fused MoE error=" << measurement.max_absolute_error
                  << " rmse=" << measurement.rmse
                  << " checksum=" << measurement.output_checksum
                  << " reference=" << measurement.reference_checksum
                  << " partial_combine=" << measurement.debug_checksum
                  << " cpu_combine=" << measurement.debug_reference_checksum
                  << " combine_error=" << measurement.debug_max_error << '\n';
        throw std::runtime_error("fused MoE failed CPU-reference validation");
      }
      const double gpu_median = median(measurement.gpu_micros);
      const double wall_median = median(measurement.wall_micros);
      const double gib_per_second = static_cast<double>(measurement.weight_bytes) /
          (gpu_median * 1e3);
      std::cout << "    {\"threadgroups\": " << measurement.threadgroups
                << ", \"threads_per_threadgroup\": "
                << measurement.threads_per_threadgroup
                << ", \"weight_bytes\": " << measurement.weight_bytes
                << ", \"two_stage\": "
                << (measurement.two_stage ? "true" : "false")
                << ", \"weight_gib_per_second\": " << gib_per_second
                << ", \"median_wall_micros\": " << wall_median
                << ", \"median_gpu_micros\": " << gpu_median
                << ", \"output_checksum\": " << measurement.output_checksum
                << ", \"reference_checksum\": " << measurement.reference_checksum
                << ", \"max_absolute_error\": "
                << measurement.max_absolute_error
                << ", \"rmse\": " << measurement.rmse
                << ", \"debug_checksum\": " << measurement.debug_checksum
                << ", \"debug_reference_checksum\": "
                << measurement.debug_reference_checksum
                << ", \"debug_max_error\": " << measurement.debug_max_error
                << ", \"trials\": ";
      print_trials(measurement.wall_micros, measurement.gpu_micros);
      std::cout << "}" << (index + 1 == options.thread_sweep.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "runnel-metal-fused-moe: " << error.what() << '\n';
    return 1;
  }
}
