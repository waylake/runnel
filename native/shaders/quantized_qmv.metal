#include <metal_stdlib>

using namespace metal;

struct QmvParams {
  uint bits;
  uint reduction_size;
  uint output_size;
  uint group_count;
  uint quant_group_size;
  uint weight_bank_output_size;
  uint rows_per_expert;
};

inline float bf16_to_float(ushort value) {
  return as_type<float>(static_cast<uint>(value) << 16);
}

inline float dot_q4(
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
  const float x_sum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
  return qx * scale + x_sum * bias;
}

inline float dot_q8(
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

kernel void qmv_weight_major(
    const device uint* weights [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant QmvParams& params [[buffer(5)]],
    const device uint* selected_experts [[buffer(6)]],
    const uint group_id [[threadgroup_position_in_grid]],
    const uint thread_id [[thread_index_in_threadgroup]],
    const uint simd_group_id [[simdgroup_index_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint threadgroup_size [[threads_per_threadgroup]]) {
  if (params.bits != 4 && params.bits != 8) {
    return;
  }

  constexpr uint max_reduction = 2048;
  threadgroup float x_tile[max_reduction];

  for (uint index = thread_id; index < params.reduction_size;
       index += threadgroup_size) {
    x_tile[index] = bf16_to_float(x[index]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint values_per_pack = 32 / params.bits;
  const uint simdgroups = threadgroup_size / 32;
  const uint rows_per_simdgroup =
      params.output_size / (params.group_count * simdgroups);
  const uint first_row =
      (group_id * simdgroups + simd_group_id) * rows_per_simdgroup;
  const uint packed_stride = params.reduction_size / values_per_pack;
  const uint quant_groups = params.reduction_size / params.quant_group_size;
  const uint reduction_start = lane_id * values_per_pack;
  const uint packs_per_iteration = values_per_pack * 32;
  uint weight_row_start = first_row;
  if (params.weight_bank_output_size != params.output_size) {
    const uint local_expert = first_row / params.rows_per_expert;
    const uint row_in_expert = first_row % params.rows_per_expert;
    weight_row_start =
        selected_experts[local_expert] * params.rows_per_expert + row_in_expert;
  }

  for (uint row_block = 0; row_block < rows_per_simdgroup; row_block += 4) {
    const uint row0 = first_row + row_block;
    const uint row1 = row0 + 1;
    const uint row2 = row0 + 2;
    const uint row3 = row0 + 3;
    const uint weight_row0 = weight_row_start + row_block;
    const uint weight_row1 = weight_row0 + 1;
    const uint weight_row2 = weight_row0 + 2;
    const uint weight_row3 = weight_row0 + 3;
    const device uint* weight0 = weights + weight_row0 * packed_stride;
    const device uint* weight1 = weights + weight_row1 * packed_stride;
    const device uint* weight2 = weights + weight_row2 * packed_stride;
    const device uint* weight3 = weights + weight_row3 * packed_stride;
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;
    float accumulator2 = 0.0f;
    float accumulator3 = 0.0f;

    for (uint reduction = reduction_start; reduction < params.reduction_size;
         reduction += packs_per_iteration) {
      const uint packed_offset = reduction / values_per_pack;
      const uint quant_group = reduction / params.quant_group_size;
      if (params.bits == 4) {
        accumulator0 += dot_q4(weight0, x_tile, reduction, packed_offset,
            bf16_to_float(scales[weight_row0 * quant_groups + quant_group]),
            bf16_to_float(biases[weight_row0 * quant_groups + quant_group]));
        accumulator1 += dot_q4(weight1, x_tile, reduction, packed_offset,
            bf16_to_float(scales[weight_row1 * quant_groups + quant_group]),
            bf16_to_float(biases[weight_row1 * quant_groups + quant_group]));
        accumulator2 += dot_q4(weight2, x_tile, reduction, packed_offset,
            bf16_to_float(scales[weight_row2 * quant_groups + quant_group]),
            bf16_to_float(biases[weight_row2 * quant_groups + quant_group]));
        accumulator3 += dot_q4(weight3, x_tile, reduction, packed_offset,
            bf16_to_float(scales[weight_row3 * quant_groups + quant_group]),
            bf16_to_float(biases[weight_row3 * quant_groups + quant_group]));
      } else {
        accumulator0 += dot_q8(weight0, x_tile, reduction, packed_offset,
            bf16_to_float(scales[weight_row0 * quant_groups + quant_group]),
            bf16_to_float(biases[weight_row0 * quant_groups + quant_group]));
        accumulator1 += dot_q8(weight1, x_tile, reduction, packed_offset,
            bf16_to_float(scales[weight_row1 * quant_groups + quant_group]),
            bf16_to_float(biases[weight_row1 * quant_groups + quant_group]));
        accumulator2 += dot_q8(weight2, x_tile, reduction, packed_offset,
            bf16_to_float(scales[weight_row2 * quant_groups + quant_group]),
            bf16_to_float(biases[weight_row2 * quant_groups + quant_group]));
        accumulator3 += dot_q8(weight3, x_tile, reduction, packed_offset,
            bf16_to_float(scales[weight_row3 * quant_groups + quant_group]),
            bf16_to_float(biases[weight_row3 * quant_groups + quant_group]));
      }
    }

    accumulator0 = simd_sum(accumulator0);
    accumulator1 = simd_sum(accumulator1);
    accumulator2 = simd_sum(accumulator2);
    accumulator3 = simd_sum(accumulator3);
    if (lane_id == 0) {
      y[row0] = accumulator0;
      y[row1] = accumulator1;
      y[row2] = accumulator2;
      y[row3] = accumulator3;
    }
  }
}
