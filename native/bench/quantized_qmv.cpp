#include "runnel/quantized_qmv_probe.hpp"

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
  std::size_t output_size = 65536;
  std::size_t reduction_size = 2048;
  std::size_t quant_group_size = 64;
  std::size_t threadgroups = 32;
  std::vector<std::size_t> thread_sweep{64, 128, 256, 512};
  std::size_t weight_bank_output_size = 0;
  std::size_t rows_per_expert = 0;
  std::vector<std::uint32_t> selected_experts;
  std::uint32_t warmups = 2;
  std::uint32_t trials = 7;
  std::string dataset_path;
};

std::string require_value(int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(argv[index]) + " requires a value");
  }
  return argv[++index];
}

std::vector<std::size_t> parse_threads(const std::string& text) {
  std::vector<std::size_t> values;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::string token = text.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start);
    if (token.empty()) {
      throw std::invalid_argument("--threads contains an empty value");
    }
    const unsigned long parsed = std::stoul(token);
    if (parsed == 0 || parsed > 1024) {
      throw std::invalid_argument("--threads values must be in [1, 1024]");
    }
    values.push_back(static_cast<std::size_t>(parsed));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return values;
}

std::vector<std::uint32_t> parse_experts(const std::string& text) {
  std::vector<std::uint32_t> values;
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
    values.push_back(static_cast<std::uint32_t>(parsed));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return values;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--bits") {
      const std::string value = require_value(argc, argv, index);
      if (value == "q4") {
        options.bits = runnel::metal::QuantizationBits::q4;
      } else if (value == "q8") {
        options.bits = runnel::metal::QuantizationBits::q8;
      } else {
        throw std::invalid_argument("--bits must be q4 or q8");
      }
    } else if (argument == "--output-size") {
      options.output_size = std::stoull(require_value(argc, argv, index));
    } else if (argument == "--reduction-size") {
      options.reduction_size = std::stoull(require_value(argc, argv, index));
    } else if (argument == "--quant-group-size") {
      options.quant_group_size =
          std::stoull(require_value(argc, argv, index));
    } else if (argument == "--groups") {
      options.threadgroups = std::stoull(require_value(argc, argv, index));
    } else if (argument == "--threads") {
      options.thread_sweep =
          parse_threads(require_value(argc, argv, index));
    } else if (argument == "--weight-bank-output-size") {
      options.weight_bank_output_size =
          std::stoull(require_value(argc, argv, index));
    } else if (argument == "--rows-per-expert") {
      options.rows_per_expert =
          std::stoull(require_value(argc, argv, index));
    } else if (argument == "--experts") {
      options.selected_experts =
          parse_experts(require_value(argc, argv, index));
    } else if (argument == "--warmups") {
      options.warmups = static_cast<std::uint32_t>(
          std::stoul(require_value(argc, argv, index)));
    } else if (argument == "--trials") {
      options.trials = static_cast<std::uint32_t>(
          std::stoul(require_value(argc, argv, index)));
    } else if (argument == "--dataset") {
      options.dataset_path = require_value(argc, argv, index);
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: runnel-metal-qmv [--bits q4|q8] "
                   "[--output-size N] [--reduction-size N] "
                   "[--quant-group-size N] [--groups N] "
                   "[--threads 64,128,256,512] "
                   "[--weight-bank-output-size N] [--rows-per-expert N] "
                   "[--experts I,J,...] [--warmups N] [--trials N] "
                   "[--dataset PATH]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  const bool has_expert_mode = options.weight_bank_output_size != 0 ||
      options.rows_per_expert != 0 || !options.selected_experts.empty();
  if (has_expert_mode &&
      (options.weight_bank_output_size == 0 || options.rows_per_expert == 0 ||
       options.selected_experts.empty())) {
    throw std::invalid_argument(
        "expert mode requires bank size, rows per expert, and expert list");
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

void print_values(const std::vector<float>& values) {
  std::cout << "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ", ";
    }
    std::cout << values[index];
  }
  std::cout << "]";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    runnel::metal::QuantizedQmvProbe probe(
        options.bits,
        options.output_size,
        options.reduction_size,
        options.quant_group_size,
        options.weight_bank_output_size,
        options.rows_per_expert,
        options.selected_experts);
    if (!options.dataset_path.empty()) {
      probe.write_dataset(options.dataset_path);
    }

    const char* bits = options.bits == runnel::metal::QuantizationBits::q4
                           ? "q4"
                           : "q8";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "{\n  \"protocol\": {\"bits\": \"" << bits
              << "\", \"output_size\": " << options.output_size
              << ", \"reduction_size\": " << options.reduction_size
              << ", \"quant_group_size\": " << options.quant_group_size
              << ", \"threadgroups\": " << options.threadgroups
              << ", \"weight_bank_output_size\": "
              << (options.weight_bank_output_size == 0
                      ? options.output_size
                      : options.weight_bank_output_size)
              << ", \"rows_per_expert\": "
              << (options.rows_per_expert == 0
                      ? options.output_size
                      : options.rows_per_expert)
              << ", \"selected_experts\": [";
    for (std::size_t index = 0; index < options.selected_experts.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << options.selected_experts[index];
    }
    std::cout << "], \"warmups\": " << options.warmups
              << ", \"trials\": " << options.trials
              << "},\n  \"measurements\": [\n";

    for (std::size_t index = 0; index < options.thread_sweep.size(); ++index) {
      const auto measurement = probe.measure(
          options.threadgroups,
          options.thread_sweep[index],
          options.warmups,
          options.trials);
      if (measurement.rmse > 1e-4 ||
          std::abs(measurement.output_checksum - measurement.reference_checksum) >
              1e-3) {
        throw std::runtime_error("quantized QMV failed CPU-reference validation");
      }
      const double gpu_median = median(measurement.gpu_micros);
      const double wall_median = median(measurement.wall_micros);
      const double gib_per_second = static_cast<double>(measurement.weight_bytes) /
          (gpu_median * 1e3);
      std::cout << "    {\"threadgroups\": " << measurement.threadgroups
                << ", \"threads_per_threadgroup\": "
                << measurement.threads_per_threadgroup
                << ", \"weight_bytes\": " << measurement.weight_bytes
                << ", \"weight_gib_per_second\": " << gib_per_second
                << ", \"median_wall_micros\": " << wall_median
                << ", \"median_gpu_micros\": " << gpu_median
                << ", \"output_checksum\": " << measurement.output_checksum
                << ", \"reference_checksum\": " << measurement.reference_checksum
                << ", \"max_reference_magnitude\": "
                << measurement.max_reference_magnitude
                << ", \"max_absolute_error\": "
                << measurement.max_absolute_error
                << ", \"max_relative_error\": "
                << measurement.max_relative_error
                << ", \"rmse\": " << measurement.rmse
                << ", \"first_actual\": ";
      print_values(measurement.first_actual);
      std::cout << ", \"first_reference\": ";
      print_values(measurement.first_reference);
      std::cout << ", \"trials\": ";
      print_trials(measurement.wall_micros, measurement.gpu_micros);
      std::cout << "}" << (index + 1 == options.thread_sweep.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "runnel-metal-qmv: " << error.what() << '\n';
    return 1;
  }
}
