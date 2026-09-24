#include <metal_stdlib>

using namespace metal;

struct QmmParams {
  uint bits;
  uint token_count;
  uint output_count;
  uint reduction_size;
  uint quant_group_count;
  uint token_tile;
  uint output_tile;
  uint reduction_tile;
};

inline float bf16_to_float(ushort value) {
  return as_type<float>(static_cast<uint>(value) << 16);
}

inline float dot_q4_tile(
    const device uint* weights,
    const threadgroup float* x,
    uint reduction,
    uint packed_offset,
    float scale,
    float bias) {
  const uint packed = weights[packed_offset];
  const float x0 = x[reduction];
  const float x1 = x[reduction + 1];
  const float x2 = x[reduction + 2];
  const float x3 = x[reduction + 3];
  const float x4 = x[reduction + 4];
  const float x5 = x[reduction + 5];
  const float x6 = x[reduction + 6];
  const float x7 = x[reduction + 7];
  const float qx =
      static_cast<float>(packed & 0x0FU) * x0 +
      static_cast<float>((packed >> 4) & 0x0FU) * x1 +
      static_cast<float>((packed >> 8) & 0x0FU) * x2 +
      static_cast<float>((packed >> 12) & 0x0FU) * x3 +
      static_cast<float>((packed >> 16) & 0x0FU) * x4 +
      static_cast<float>((packed >> 20) & 0x0FU) * x5 +
      static_cast<float>((packed >> 24) & 0x0FU) * x6 +
      static_cast<float>((packed >> 28) & 0x0FU) * x7;
  return qx * scale + (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) * bias;
}

inline float dot_q8_tile(
    const device uint* weights,
    const threadgroup float* x,
    uint reduction,
    uint packed_offset,
    float scale,
    float bias) {
  const uint packed = weights[packed_offset];
  const float x0 = x[reduction];
  const float x1 = x[reduction + 1];
  const float x2 = x[reduction + 2];
  const float x3 = x[reduction + 3];
  const float qx =
      static_cast<float>(packed & 0xFFU) * x0 +
      static_cast<float>((packed >> 8) & 0xFFU) * x1 +
      static_cast<float>((packed >> 16) & 0xFFU) * x2 +
      static_cast<float>((packed >> 24) & 0xFFU) * x3;
  return qx * scale + (x0 + x1 + x2 + x3) * bias;
}

kernel void qmm_4d_tile(
    const device uint* weights [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant QmmParams& params [[buffer(5)]],
    uint3 tile_id [[threadgroup_position_in_grid]],
    const uint thread_id [[thread_index_in_threadgroup]],
    const uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]]) {
  if (params.bits != 4 && params.bits != 8) {
    return;
  }

  const uint threadgroup_size = 256;
  constexpr uint token_tile = 8;
  constexpr uint output_tile = 32;
  constexpr uint reduction_tile = 256;
  threadgroup float x_tile[token_tile * reduction_tile];
  const uint token_base = tile_id.y * token_tile;
  const uint output_base = tile_id.x * output_tile;
  const uint values_per_pack = 32 / params.bits;
  const uint packed_stride = params.reduction_size / values_per_pack;
  const uint quant_group_stride = params.quant_group_count;

  float accumulators[4][token_tile] = {};
  for (uint reduction_base = 0; reduction_base < params.reduction_size;
       reduction_base += reduction_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = thread_id; index < token_tile * reduction_tile;
         index += threadgroup_size) {
      const uint token = index / reduction_tile;
      const uint reduction = index % reduction_tile;
      x_tile[index] = bf16_to_float(
          x[(token_base + token) * params.reduction_size + reduction_base + reduction]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint local_row = 0; local_row < 4; ++local_row) {
      const uint row = output_base + simdgroup_id * 4 + local_row;
      for (uint reduction = lane_id * values_per_pack;
           reduction < reduction_tile;
           reduction += 32 * values_per_pack) {
        const uint packed_offset = (reduction_base + reduction) / values_per_pack;
        const uint quant_group = (reduction_base + reduction) / 64;
        const float scale = bf16_to_float(scales[row * quant_group_stride + quant_group]);
        const float bias = bf16_to_float(biases[row * quant_group_stride + quant_group]);
        if (params.bits == 4) {
          for (uint token = 0; token < token_tile; ++token) {
            accumulators[local_row][token] += dot_q4_tile(
                weights + row * packed_stride,
                x_tile + token * reduction_tile,
                reduction,
                packed_offset,
                scale,
                bias);
          }
        } else {
          for (uint token = 0; token < token_tile; ++token) {
            accumulators[local_row][token] += dot_q8_tile(
                weights + row * packed_stride,
                x_tile + token * reduction_tile,
                reduction,
                packed_offset,
                scale,
                bias);
          }
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint local_row = 0; local_row < 4; ++local_row) {
    const uint row = output_base + simdgroup_id * 4 + local_row;
    for (uint token = 0; token < token_tile; ++token) {
      const float value = simd_sum(accumulators[local_row][token]);
      if (lane_id == 0) {
        y[(token_base + token) * params.output_count + row] = value;
      }
    }
  }
}
