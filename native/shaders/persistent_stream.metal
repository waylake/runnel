#include <metal_stdlib>

using namespace metal;

struct StreamParams {
  uint total_vectors;
  uint group_count;
  uint vectors_per_group;
  uint thread_count;
};

kernel void stream_persistent(
    const device uint4* weights [[buffer(0)]],
    device uint* checksum [[buffer(1)]],
    constant StreamParams& params [[buffer(2)]],
    const uint group_id [[threadgroup_position_in_grid]],
    const uint thread_id [[thread_index_in_threadgroup]],
    const uint simd_group_id [[simdgroup_index_in_threadgroup]],
    const uint simd_size [[threads_per_simdgroup]],
    const uint threadgroup_size [[threads_per_threadgroup]]) {
  threadgroup uint partial_sum[32];

  const uint thread_count = params.thread_count;
  const uint vectors_per_group = params.vectors_per_group;
  const uint group_start = group_id * vectors_per_group;
  const uint group_end = group_start + vectors_per_group;
  uint local_index = thread_id;
  uint accumulator0 = 0;
  uint accumulator1 = 0;
  uint accumulator2 = 0;
  uint accumulator3 = 0;
  uint accumulator4 = 0;
  uint accumulator5 = 0;
  uint accumulator6 = 0;
  uint accumulator7 = 0;

  constexpr uint unroll = 8;
  const uint unrolled_local_end =
      (vectors_per_group / (thread_count * unroll)) * thread_count * unroll;

  while (local_index + (unroll - 1) * thread_count < unrolled_local_end) {
    const uint base = group_start + local_index;
    uint4 v0 = weights[base];
    uint4 v1 = weights[base + thread_count];
    uint4 v2 = weights[base + 2 * thread_count];
    uint4 v3 = weights[base + 3 * thread_count];
    uint4 v4 = weights[base + 4 * thread_count];
    uint4 v5 = weights[base + 5 * thread_count];
    uint4 v6 = weights[base + 6 * thread_count];
    uint4 v7 = weights[base + 7 * thread_count];
    accumulator0 += v0.x;
    accumulator1 += v1.x;
    accumulator2 += v2.x;
    accumulator3 += v3.x;
    accumulator4 += v4.x;
    accumulator5 += v5.x;
    accumulator6 += v6.x;
    accumulator7 += v7.x;
    local_index += unroll * thread_count;
  }

  uint accumulator = accumulator0 + accumulator1 + accumulator2 + accumulator3 +
      accumulator4 + accumulator5 + accumulator6 + accumulator7;
  while (local_index < vectors_per_group) {
    accumulator += weights[group_start + local_index].x;
    local_index += thread_count;
  }

  partial_sum[simd_group_id] = simd_sum(accumulator);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (thread_id == 0) {
    uint total = 0;
    for (uint index = 0; index < threadgroup_size / simd_size; ++index) {
      total += partial_sum[index];
    }
    checksum[group_id] = total;
  }
}
