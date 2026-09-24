#include "runnel/metal_probe.hpp"

#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef RUNNEL_DEFAULT_METAL_SHADER
#define RUNNEL_DEFAULT_METAL_SHADER "shaders/persistent_stream.metal"
#endif

namespace runnel::metal {
namespace {

using Clock = std::chrono::steady_clock;

struct alignas(16) StreamParams {
  std::uint32_t total_vectors;
  std::uint32_t group_count;
  std::uint32_t vectors_per_group;
  std::uint32_t thread_count;
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

std::uint32_t gpu_core_count() {
  io_service_t service = IOServiceGetMatchingService(
      kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
  if (service == IO_OBJECT_NULL) {
    return 0;
  }

  CFTypeRef value = IORegistryEntryCreateCFProperty(
      service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
  std::uint32_t count = 0;
  if (value != nullptr && CFGetTypeID(value) == CFNumberGetTypeID()) {
    static_cast<void>(CFNumberGetValue(
        static_cast<CFNumberRef>(value), kCFNumberSInt32Type, &count));
  }
  if (value != nullptr) {
    CFRelease(value);
  }
  IOObjectRelease(service);
  return count;
}

}  // namespace

class StreamProbe::Impl {
 public:
  explicit Impl(std::size_t buffer_bytes) : buffer_bytes_(buffer_bytes) {
    if (buffer_bytes == 0 || buffer_bytes % 4096 != 0) {
      fail("buffer size must be a non-zero multiple of 4096 bytes");
    }
    if (buffer_bytes / 16 > std::numeric_limits<std::uint32_t>::max()) {
      fail("buffer is too large for the probe's 32-bit vector indexing");
    }

    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) {
      fail("Metal is unavailable on this machine");
    }

    NSError* error = nil;
    NSString* source = [NSString
        stringWithContentsOfFile:@(RUNNEL_DEFAULT_METAL_SHADER)
                       encoding:NSUTF8StringEncoding
                          error:&error];
    if (source == nil) {
      fail(ns_error(error, "could not read the persistent-stream shader"));
    }

    id<MTLLibrary> library = [device_ newLibraryWithSource:source
                                                     options:nil
                                                       error:&error];
    if (library == nil) {
      fail(ns_error(error, "could not compile the persistent-stream shader"));
    }

    id<MTLFunction> function =
        [library newFunctionWithName:@"stream_persistent"];
    if (function == nil) {
      fail("persistent-stream shader is missing stream_persistent");
    }

    pipeline_ = [device_ newComputePipelineStateWithFunction:function
                                                       error:&error];
    if (pipeline_ == nil) {
      fail(ns_error(error, "could not create the persistent-stream pipeline"));
    }

    queue_ = [device_ newCommandQueue];
    if (queue_ == nil) {
      fail("could not create a Metal command queue");
    }

    weights_ = [device_ newBufferWithLength:buffer_bytes_
                                     options:MTLResourceStorageModeShared];
    checksum_ = [device_ newBufferWithLength:256
                                     options:MTLResourceStorageModeShared];
    parameters_ = [device_ newBufferWithLength:sizeof(StreamParams)
                                       options:MTLResourceStorageModeShared];
    if (weights_ == nil || checksum_ == nil || parameters_ == nil) {
      fail("could not allocate Metal probe buffers");
    }

    // Commit every page before timing and use a distinct deterministic word
    // pattern per page. Constant-fill buffers are needlessly compressible and
    // make a zero checksum ambiguous.
    constexpr std::size_t page_bytes = 4096;
    auto* words = static_cast<std::uint32_t*>(weights_.contents);
    const std::size_t words_per_page = page_bytes / sizeof(std::uint32_t);
    const std::size_t page_count = buffer_bytes_ / page_bytes;
    for (std::size_t page = 0; page < page_count; ++page) {
      const std::uint32_t page_seed = static_cast<std::uint32_t>(page) + 1U;
      for (std::size_t index = 0; index < words_per_page; ++index) {
        words[page * words_per_page + index] =
            page_seed + static_cast<std::uint32_t>(index * 0x9E3779B9U);
      }
    }
    std::memset(checksum_.contents, 0, checksum_.length);
  }

  DeviceInfo device_info() const {
    DeviceInfo info;
    const char* device_name = device_.name.UTF8String;
    info.name = device_name == nullptr ? "unknown" : device_name;
    info.core_count = gpu_core_count();
    info.max_threads_per_threadgroup = pipeline_.maxTotalThreadsPerThreadgroup;
    info.max_threads_per_warp = pipeline_.threadExecutionWidth;
    info.recommended_working_set_bytes = device_.recommendedMaxWorkingSetSize;
    info.low_power = device_.lowPower;
    info.headless = device_.headless;
    info.has_unified_memory = device_.hasUnifiedMemory;
    return info;
  }

  StreamMeasurement measure(
      std::size_t threadgroups,
      std::size_t threads_per_threadgroup,
      std::uint32_t warmups,
      std::uint32_t trials) const {
    if (threadgroups == 0 || threads_per_threadgroup == 0) {
      fail("threadgroup and thread counts must be positive");
    }
    if (threadgroups > std::numeric_limits<std::uint32_t>::max()) {
      fail("threadgroup count exceeds the probe's 32-bit metadata limit");
    }
    if (threads_per_threadgroup > pipeline_.maxTotalThreadsPerThreadgroup) {
      std::ostringstream stream;
      stream << "requested " << threads_per_threadgroup
             << " threads/threadgroup, device maximum is "
             << pipeline_.maxTotalThreadsPerThreadgroup;
      fail(stream.str());
    }

    const std::size_t total_vectors = buffer_bytes_ / 16;
    if (total_vectors % threadgroups != 0) {
      fail("threadgroups must evenly divide the uint4 vector count");
    }
    const std::size_t vectors_per_group = total_vectors / threadgroups;

    StreamParams params{
        static_cast<std::uint32_t>(total_vectors),
        static_cast<std::uint32_t>(threadgroups),
        static_cast<std::uint32_t>(vectors_per_group),
        static_cast<std::uint32_t>(threads_per_threadgroup),
    };
    std::memcpy(parameters_.contents, &params, sizeof(params));

    const auto run_once = [&]() {
      std::memset(checksum_.contents, 0, checksum_.length);
      id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder =
          [command_buffer computeCommandEncoder];
      [encoder setComputePipelineState:pipeline_];
      [encoder setBuffer:weights_ offset:0 atIndex:0];
      [encoder setBuffer:checksum_ offset:0 atIndex:1];
      [encoder setBuffer:parameters_ offset:0 atIndex:2];
      [encoder dispatchThreadgroups:MTLSizeMake(threadgroups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
      [encoder endEncoding];
      [command_buffer commit];

      const auto started = Clock::now();
      [command_buffer waitUntilCompleted];
      const auto ended = Clock::now();
      if (command_buffer.error != nil) {
        fail(ns_error(command_buffer.error, "Metal stream probe failed"));
      }
      if (command_buffer.GPUStartTime == 0 || command_buffer.GPUEndTime == 0) {
        fail("Metal did not provide GPU timing for the stream probe");
      }

      StreamTrial trial;
      trial.wall_micros =
          std::chrono::duration<double, std::micro>(ended - started).count();
      trial.gpu_micros =
          (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1e6;
      return trial;
    };

    for (std::uint32_t index = 0; index < warmups; ++index) {
      static_cast<void>(run_once());
    }

    StreamMeasurement measurement;
    measurement.bytes = buffer_bytes_;
    measurement.threadgroups = threadgroups;
    measurement.threads_per_threadgroup = threads_per_threadgroup;
    measurement.trials.reserve(trials);
    for (std::uint32_t index = 0; index < trials; ++index) {
      measurement.trials.push_back(run_once());
    }
    const auto* checksum_words =
        static_cast<const std::uint32_t*>(checksum_.contents);
    measurement.checksum = 0;
    for (std::size_t group = 0; group < threadgroups; ++group) {
      measurement.checksum += checksum_words[group];
    }
    return measurement;
  }

 private:
  std::size_t buffer_bytes_{};
  __strong id<MTLDevice> device_{};
  __strong id<MTLCommandQueue> queue_{};
  __strong id<MTLComputePipelineState> pipeline_{};
  __strong id<MTLBuffer> weights_{};
  __strong id<MTLBuffer> checksum_{};
  __strong id<MTLBuffer> parameters_{};
};

StreamProbe::StreamProbe(std::size_t buffer_bytes)
    : impl_(std::make_unique<Impl>(buffer_bytes)) {}

StreamProbe::~StreamProbe() = default;
StreamProbe::StreamProbe(StreamProbe&&) noexcept = default;
StreamProbe& StreamProbe::operator=(StreamProbe&&) noexcept = default;

DeviceInfo StreamProbe::device_info() const {
  return impl_->device_info();
}

StreamMeasurement StreamProbe::measure(
    std::size_t threadgroups,
    std::size_t threads_per_threadgroup,
    std::uint32_t warmups,
    std::uint32_t trials) const {
  return impl_->measure(
      threadgroups, threads_per_threadgroup, warmups, trials);
}

}  // namespace runnel::metal
