#include <metal_stdlib>

using namespace metal;

struct FusedMoeParams {
  uint bits;
  uint group_count;
  uint expert_count;
  uint simdgroups_per_expert;
  uint channels_per_simdgroup;
  uint hidden_size;
  uint intermediate_size;
  uint debug_stage;
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
  return qx * scale +
      (x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) * bias;
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

template <int bits>
inline void gate_up_impl(
    const device uint* weights,
    const device ushort* scales,
    const device ushort* biases,
    const threadgroup float* x,
    threadgroup float* up_values,
    threadgroup float* gate_values,
    uint expert_row_base,
    uint channel_base,
    uint channels,
    uint lane_id,
    constant FusedMoeParams& params) {
  constexpr uint hidden_size = 2048;
  constexpr uint intermediate_size = 512;
  constexpr uint values_per_pack = 32 / bits;
  const uint packed_stride = hidden_size / values_per_pack;
  const uint quant_groups = hidden_size / 64;
  const uint reduction_start = lane_id * values_per_pack;
  const uint reduction_stride = values_per_pack * 32;
  const uint gate_row_offset = intermediate_size;

  for (uint channel_block = 0; channel_block < channels; channel_block += 4) {
    const uint up_row0 = expert_row_base + channel_base + channel_block;
    const uint gate_row0 = up_row0 + gate_row_offset;
    const device uint* up_weights0 = weights + up_row0 * packed_stride;
    const device uint* up_weights1 = up_weights0 + packed_stride;
    const device uint* up_weights2 = up_weights0 + 2 * packed_stride;
    const device uint* up_weights3 = up_weights0 + 3 * packed_stride;
    const device uint* gate_weights0 = weights + gate_row0 * packed_stride;
    const device uint* gate_weights1 = gate_weights0 + packed_stride;
    const device uint* gate_weights2 = gate_weights0 + 2 * packed_stride;
    const device uint* gate_weights3 = gate_weights0 + 3 * packed_stride;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float up2 = 0.0f;
    float up3 = 0.0f;
    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float gate2 = 0.0f;
    float gate3 = 0.0f;

    for (uint reduction = reduction_start; reduction < hidden_size;
         reduction += reduction_stride) {
      const uint packed_offset = reduction / values_per_pack;
      const uint quant_group = reduction / 64;
      if (bits == 4) {
        const float us0 = bf16_to_float(scales[(up_row0) * quant_groups + quant_group]);
        const float us1 = bf16_to_float(scales[(up_row0 + 1) * quant_groups + quant_group]);
        const float us2 = bf16_to_float(scales[(up_row0 + 2) * quant_groups + quant_group]);
        const float us3 = bf16_to_float(scales[(up_row0 + 3) * quant_groups + quant_group]);
        const float ub0 = bf16_to_float(biases[(up_row0) * quant_groups + quant_group]);
        const float ub1 = bf16_to_float(biases[(up_row0 + 1) * quant_groups + quant_group]);
        const float ub2 = bf16_to_float(biases[(up_row0 + 2) * quant_groups + quant_group]);
        const float ub3 = bf16_to_float(biases[(up_row0 + 3) * quant_groups + quant_group]);
        const float gs0 = bf16_to_float(scales[(gate_row0) * quant_groups + quant_group]);
        const float gs1 = bf16_to_float(scales[(gate_row0 + 1) * quant_groups + quant_group]);
        const float gs2 = bf16_to_float(scales[(gate_row0 + 2) * quant_groups + quant_group]);
        const float gs3 = bf16_to_float(scales[(gate_row0 + 3) * quant_groups + quant_group]);
        const float gb0 = bf16_to_float(biases[(gate_row0) * quant_groups + quant_group]);
        const float gb1 = bf16_to_float(biases[(gate_row0 + 1) * quant_groups + quant_group]);
        const float gb2 = bf16_to_float(biases[(gate_row0 + 2) * quant_groups + quant_group]);
        const float gb3 = bf16_to_float(biases[(gate_row0 + 3) * quant_groups + quant_group]);
        up0 += dot_q4(up_weights0, x, reduction, packed_offset, us0, ub0);
        up1 += dot_q4(up_weights1, x, reduction, packed_offset, us1, ub1);
        up2 += dot_q4(up_weights2, x, reduction, packed_offset, us2, ub2);
        up3 += dot_q4(up_weights3, x, reduction, packed_offset, us3, ub3);
        gate0 += dot_q4(gate_weights0, x, reduction, packed_offset, gs0, gb0);
        gate1 += dot_q4(gate_weights1, x, reduction, packed_offset, gs1, gb1);
        gate2 += dot_q4(gate_weights2, x, reduction, packed_offset, gs2, gb2);
        gate3 += dot_q4(gate_weights3, x, reduction, packed_offset, gs3, gb3);
      } else {
        const float us0 = bf16_to_float(scales[(up_row0) * quant_groups + quant_group]);
        const float us1 = bf16_to_float(scales[(up_row0 + 1) * quant_groups + quant_group]);
        const float us2 = bf16_to_float(scales[(up_row0 + 2) * quant_groups + quant_group]);
        const float us3 = bf16_to_float(scales[(up_row0 + 3) * quant_groups + quant_group]);
        const float ub0 = bf16_to_float(biases[(up_row0) * quant_groups + quant_group]);
        const float ub1 = bf16_to_float(biases[(up_row0 + 1) * quant_groups + quant_group]);
        const float ub2 = bf16_to_float(biases[(up_row0 + 2) * quant_groups + quant_group]);
        const float ub3 = bf16_to_float(biases[(up_row0 + 3) * quant_groups + quant_group]);
        const float gs0 = bf16_to_float(scales[(gate_row0) * quant_groups + quant_group]);
        const float gs1 = bf16_to_float(scales[(gate_row0 + 1) * quant_groups + quant_group]);
        const float gs2 = bf16_to_float(scales[(gate_row0 + 2) * quant_groups + quant_group]);
        const float gs3 = bf16_to_float(scales[(gate_row0 + 3) * quant_groups + quant_group]);
        const float gb0 = bf16_to_float(biases[(gate_row0) * quant_groups + quant_group]);
        const float gb1 = bf16_to_float(biases[(gate_row0 + 1) * quant_groups + quant_group]);
        const float gb2 = bf16_to_float(biases[(gate_row0 + 2) * quant_groups + quant_group]);
        const float gb3 = bf16_to_float(biases[(gate_row0 + 3) * quant_groups + quant_group]);
        up0 += dot_q8(up_weights0, x, reduction, packed_offset, us0, ub0);
        up1 += dot_q8(up_weights1, x, reduction, packed_offset, us1, ub1);
        up2 += dot_q8(up_weights2, x, reduction, packed_offset, us2, ub2);
        up3 += dot_q8(up_weights3, x, reduction, packed_offset, us3, ub3);
        gate0 += dot_q8(gate_weights0, x, reduction, packed_offset, gs0, gb0);
        gate1 += dot_q8(gate_weights1, x, reduction, packed_offset, gs1, gb1);
        gate2 += dot_q8(gate_weights2, x, reduction, packed_offset, gs2, gb2);
        gate3 += dot_q8(gate_weights3, x, reduction, packed_offset, gs3, gb3);
      }
    }

    up0 = simd_sum(up0);
    up1 = simd_sum(up1);
    up2 = simd_sum(up2);
    up3 = simd_sum(up3);
    gate0 = simd_sum(gate0);
    gate1 = simd_sum(gate1);
    gate2 = simd_sum(gate2);
    gate3 = simd_sum(gate3);
    if (lane_id == 0) {
      const uint output_index = channel_block;
      up_values[output_index] = up0;
      up_values[output_index + 1] = up1;
      up_values[output_index + 2] = up2;
      up_values[output_index + 3] = up3;
      gate_values[output_index] = gate0;
      gate_values[output_index + 1] = gate1;
      gate_values[output_index + 2] = gate2;
      gate_values[output_index + 3] = gate3;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float shard_dot_q4(
    const device uint* row_weights,
    const threadgroup float* intermediate,
    uint input_base,
    uint value,
    const device ushort* scales,
    const device ushort* biases,
    uint row,
    uint quant_groups) {
  const uint packed = row_weights[value / 8];
  const uint quantized = (packed >> (4 * (value % 8))) & 0x0FU;
  const uint group = (input_base + value) / 64;
  const float scale = bf16_to_float(scales[row * quant_groups + group]);
  const float bias = bf16_to_float(biases[row * quant_groups + group]);
  return quantized * scale * intermediate[input_base + value] +
      bias * intermediate[input_base + value];
}

inline float shard_dot_q8(
    const device uint* row_weights,
    const threadgroup float* intermediate,
    uint input_base,
    uint value,
    const device ushort* scales,
    const device ushort* biases,
    uint row,
    uint quant_groups) {
  const uint packed = row_weights[value / 4];
  const uint quantized = (packed >> (8 * (value % 4))) & 0xFFU;
  const uint group = (input_base + value) / 64;
  const float scale = bf16_to_float(scales[row * quant_groups + group]);
  const float bias = bf16_to_float(biases[row * quant_groups + group]);
  return quantized * scale * intermediate[input_base + value] +
      bias * intermediate[input_base + value];
}

template <int bits>
inline void down_impl(
    const device uint* weights,
    const device ushort* scales,
    const device ushort* biases,
    const threadgroup float* intermediate,
    device float* partial,
    uint expert_id,
    uint expert_slot,
    uint local_simdgroup,
    uint storage_simdgroup,
    uint channels,
    uint lane_id,
    constant FusedMoeParams& params) {
  constexpr uint hidden_size = 2048;
  constexpr uint down_size = 512;
  constexpr uint values_per_pack = 32 / bits;
  const uint packed_stride = down_size / values_per_pack;
  const uint quant_groups = down_size / 64;
  const uint rows_per_simdgroup = hidden_size;
  const uint first_row = expert_id * hidden_size;
  const uint input_base = storage_simdgroup * channels;
  const uint values_per_lane = (channels + 31) / 32;
  const uint partial_base =
      (expert_slot * params.simdgroups_per_expert + local_simdgroup) * hidden_size;

  for (uint row_block = 0; row_block < rows_per_simdgroup; row_block += 4) {
    const uint row0 = first_row + row_block;
    const uint row1 = row0 + 1;
    const uint row2 = row0 + 2;
    const uint row3 = row0 + 3;
    const device uint* weight0 = weights + row0 * packed_stride;
    const device uint* weight1 = weights + row1 * packed_stride;
    const device uint* weight2 = weights + row2 * packed_stride;
    const device uint* weight3 = weights + row3 * packed_stride;
    float value0 = 0.0f;
    float value1 = 0.0f;
    float value2 = 0.0f;
    float value3 = 0.0f;

    for (uint value_base = 0; value_base < channels;
         value_base += values_per_lane * 32) {
      const uint value_start = value_base + lane_id * values_per_lane;
      for (uint offset = 0; offset < values_per_lane; ++offset) {
        const uint value = value_start + offset;
        if (value < channels) {
          if (bits == 4) {
            value0 += shard_dot_q4(weight0, intermediate, input_base, value, scales, biases, row0, quant_groups);
            value1 += shard_dot_q4(weight1, intermediate, input_base, value, scales, biases, row1, quant_groups);
            value2 += shard_dot_q4(weight2, intermediate, input_base, value, scales, biases, row2, quant_groups);
            value3 += shard_dot_q4(weight3, intermediate, input_base, value, scales, biases, row3, quant_groups);
          } else {
            value0 += shard_dot_q8(weight0, intermediate, input_base, value, scales, biases, row0, quant_groups);
            value1 += shard_dot_q8(weight1, intermediate, input_base, value, scales, biases, row1, quant_groups);
            value2 += shard_dot_q8(weight2, intermediate, input_base, value, scales, biases, row2, quant_groups);
            value3 += shard_dot_q8(weight3, intermediate, input_base, value, scales, biases, row3, quant_groups);
          }
        }
      }
    }

    value0 = simd_sum(value0);
    value1 = simd_sum(value1);
    value2 = simd_sum(value2);
    value3 = simd_sum(value3);
    if (lane_id == 0) {
      partial[partial_base + row_block] = value0;
      partial[partial_base + row_block + 1] = value1;
      partial[partial_base + row_block + 2] = value2;
      partial[partial_base + row_block + 3] = value3;
    }
  }
}

kernel void fused_moe_layer(
    const device uint* gate_up_weights [[buffer(0)]],
    const device ushort* gate_up_scales [[buffer(1)]],
    const device ushort* gate_up_biases [[buffer(2)]],
    const device uint* down_weights [[buffer(3)]],
    const device ushort* down_scales [[buffer(4)]],
    const device ushort* down_biases [[buffer(5)]],
    const device ushort* x [[buffer(6)]],
    const device uint* selected_experts [[buffer(7)]],
    device float* partial [[buffer(8)]],
    constant FusedMoeParams& params [[buffer(9)]],
    const uint group_id [[threadgroup_position_in_grid]],
    const uint thread_id [[thread_index_in_threadgroup]],
    const uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint threadgroup_size [[threads_per_threadgroup]]) {
  if (params.bits != 4 && params.bits != 8) {
    return;
  }

  threadgroup float x_tile[2048];
  threadgroup float up_values[512];
  threadgroup float gate_values[512];
  for (uint index = thread_id; index < 2048; index += threadgroup_size) {
    x_tile[index] = bf16_to_float(x[index]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint simdgroups_per_threadgroup = threadgroup_size / 32;
  const uint global_simdgroup =
      group_id * simdgroups_per_threadgroup + simdgroup_id;
  const uint expert_slot = global_simdgroup / params.simdgroups_per_expert;
  const uint local_simdgroup =
      global_simdgroup % params.simdgroups_per_expert;
  const uint expert_id = selected_experts[expert_slot];
  const uint channel_base = local_simdgroup * params.channels_per_simdgroup;
  const uint expert_row_base = expert_id * 1024;
  const uint value_base = simdgroup_id * params.channels_per_simdgroup;

  if (params.bits == 4) {
    gate_up_impl<4>(
        gate_up_weights, gate_up_scales, gate_up_biases, x_tile,
        up_values + value_base, gate_values + value_base,
        expert_row_base, channel_base, params.channels_per_simdgroup,
        lane_id, params);
  } else {
    gate_up_impl<8>(
        gate_up_weights, gate_up_scales, gate_up_biases, x_tile,
        up_values + value_base, gate_values + value_base,
        expert_row_base, channel_base, params.channels_per_simdgroup,
        lane_id, params);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane_id < params.channels_per_simdgroup) {
    const uint index = value_base + lane_id;
    const float gate = gate_values[index];
    up_values[index] = (gate / (1.0f + metal::exp(-gate))) * up_values[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (params.debug_stage == 1) {
    if (lane_id < params.channels_per_simdgroup) {
      partial[expert_id * 512 + local_simdgroup * params.channels_per_simdgroup + lane_id] =
          up_values[value_base + lane_id];
    }
    return;
  }

  if (params.bits == 4) {
    down_impl<4>(
        down_weights, down_scales, down_biases, up_values, partial,
        expert_id, expert_slot, local_simdgroup, simdgroup_id,
        params.channels_per_simdgroup, lane_id, params);
  } else {
    down_impl<8>(
        down_weights, down_scales, down_biases, up_values, partial,
        expert_id, expert_slot, local_simdgroup, simdgroup_id,
        params.channels_per_simdgroup, lane_id, params);
  }
}

kernel void combine_moe(
    const device float* partial [[buffer(0)]],
    const device float* router_weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant FusedMoeParams& params [[buffer(3)]],
    const uint gid [[threadgroup_position_in_grid]],
    const uint lid [[thread_index_in_threadgroup]]) {
  const uint index = gid * 256 + lid;
  if (index >= params.hidden_size) {
    return;
  }
  float sum = 0.0f;
  for (uint expert = 0; expert < params.expert_count; ++expert) {
    float expert_sum = 0.0f;
    for (uint shard = 0; shard < params.simdgroups_per_expert; ++shard) {
      expert_sum += partial[(expert * params.simdgroups_per_expert + shard) *
          params.hidden_size + index];
    }
    sum += router_weights[expert] * expert_sum;
  }
  output[index] = sum;
}

kernel void gate_up_swiglu(
    const device uint* gate_up_weights [[buffer(0)]],
    const device ushort* gate_up_scales [[buffer(1)]],
    const device ushort* gate_up_biases [[buffer(2)]],
    const device ushort* x [[buffer(3)]],
    const device uint* selected_experts [[buffer(4)]],
    device float* intermediate [[buffer(5)]],
    constant FusedMoeParams& params [[buffer(6)]],
    const uint group_id [[threadgroup_position_in_grid]],
    const uint thread_id [[thread_index_in_threadgroup]],
    const uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint threadgroup_size [[threads_per_threadgroup]]) {
  if (params.bits != 4 && params.bits != 8) {
    return;
  }

  threadgroup float x_tile[2048];
  for (uint index = thread_id; index < 2048; index += threadgroup_size) {
    x_tile[index] = bf16_to_float(x[index]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint simdgroups_per_threadgroup = threadgroup_size / 32;
  const uint global_simdgroup =
      group_id * simdgroups_per_threadgroup + simdgroup_id;
  const uint expert_slot = global_simdgroup / params.simdgroups_per_expert;
  const uint local_simdgroup =
      global_simdgroup % params.simdgroups_per_expert;
  const uint expert_id = selected_experts[expert_slot];
  const uint channel_base = local_simdgroup * params.channels_per_simdgroup;
  const uint expert_row_base = expert_id * 1024;
  const uint values_per_pack = 32 / params.bits;
  const uint packed_stride = 2048 / values_per_pack;
  const uint quant_groups = 32;
  const uint reduction_start = lane_id * values_per_pack;
  const uint reduction_stride = values_per_pack * 32;

  for (uint channel_block = 0;
       channel_block < params.channels_per_simdgroup;
       channel_block += 4) {
    const uint up_row0 = expert_row_base + channel_base + channel_block;
    const uint gate_row0 = up_row0 + 512;
    const device uint* up_weights0 = gate_up_weights + up_row0 * packed_stride;
    const device uint* up_weights1 = up_weights0 + packed_stride;
    const device uint* up_weights2 = up_weights0 + 2 * packed_stride;
    const device uint* up_weights3 = up_weights0 + 3 * packed_stride;
    const device uint* gate_weights0 = gate_up_weights + gate_row0 * packed_stride;
    const device uint* gate_weights1 = gate_weights0 + packed_stride;
    const device uint* gate_weights2 = gate_weights0 + 2 * packed_stride;
    const device uint* gate_weights3 = gate_weights0 + 3 * packed_stride;
    float up0 = 0.0f;
    float up1 = 0.0f;
    float up2 = 0.0f;
    float up3 = 0.0f;
    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float gate2 = 0.0f;
    float gate3 = 0.0f;

    for (uint reduction = reduction_start; reduction < 2048;
         reduction += reduction_stride) {
      const uint packed_offset = reduction / values_per_pack;
      const uint quant_group = reduction / 64;
      if (params.bits == 4) {
        const float us0 = bf16_to_float(gate_up_scales[(up_row0) * quant_groups + quant_group]);
        const float us1 = bf16_to_float(gate_up_scales[(up_row0 + 1) * quant_groups + quant_group]);
        const float us2 = bf16_to_float(gate_up_scales[(up_row0 + 2) * quant_groups + quant_group]);
        const float us3 = bf16_to_float(gate_up_scales[(up_row0 + 3) * quant_groups + quant_group]);
        const float ub0 = bf16_to_float(gate_up_biases[(up_row0) * quant_groups + quant_group]);
        const float ub1 = bf16_to_float(gate_up_biases[(up_row0 + 1) * quant_groups + quant_group]);
        const float ub2 = bf16_to_float(gate_up_biases[(up_row0 + 2) * quant_groups + quant_group]);
        const float ub3 = bf16_to_float(gate_up_biases[(up_row0 + 3) * quant_groups + quant_group]);
        const float gs0 = bf16_to_float(gate_up_scales[(gate_row0) * quant_groups + quant_group]);
        const float gs1 = bf16_to_float(gate_up_scales[(gate_row0 + 1) * quant_groups + quant_group]);
        const float gs2 = bf16_to_float(gate_up_scales[(gate_row0 + 2) * quant_groups + quant_group]);
        const float gs3 = bf16_to_float(gate_up_scales[(gate_row0 + 3) * quant_groups + quant_group]);
        const float gb0 = bf16_to_float(gate_up_biases[(gate_row0) * quant_groups + quant_group]);
        const float gb1 = bf16_to_float(gate_up_biases[(gate_row0 + 1) * quant_groups + quant_group]);
        const float gb2 = bf16_to_float(gate_up_biases[(gate_row0 + 2) * quant_groups + quant_group]);
        const float gb3 = bf16_to_float(gate_up_biases[(gate_row0 + 3) * quant_groups + quant_group]);
        up0 += dot_q4(up_weights0, x_tile, reduction, packed_offset, us0, ub0);
        up1 += dot_q4(up_weights1, x_tile, reduction, packed_offset, us1, ub1);
        up2 += dot_q4(up_weights2, x_tile, reduction, packed_offset, us2, ub2);
        up3 += dot_q4(up_weights3, x_tile, reduction, packed_offset, us3, ub3);
        gate0 += dot_q4(gate_weights0, x_tile, reduction, packed_offset, gs0, gb0);
        gate1 += dot_q4(gate_weights1, x_tile, reduction, packed_offset, gs1, gb1);
        gate2 += dot_q4(gate_weights2, x_tile, reduction, packed_offset, gs2, gb2);
        gate3 += dot_q4(gate_weights3, x_tile, reduction, packed_offset, gs3, gb3);
      } else {
        const float us0 = bf16_to_float(gate_up_scales[(up_row0) * quant_groups + quant_group]);
        const float us1 = bf16_to_float(gate_up_scales[(up_row0 + 1) * quant_groups + quant_group]);
        const float us2 = bf16_to_float(gate_up_scales[(up_row0 + 2) * quant_groups + quant_group]);
        const float us3 = bf16_to_float(gate_up_scales[(up_row0 + 3) * quant_groups + quant_group]);
        const float ub0 = bf16_to_float(gate_up_biases[(up_row0) * quant_groups + quant_group]);
        const float ub1 = bf16_to_float(gate_up_biases[(up_row0 + 1) * quant_groups + quant_group]);
        const float ub2 = bf16_to_float(gate_up_biases[(up_row0 + 2) * quant_groups + quant_group]);
        const float ub3 = bf16_to_float(gate_up_biases[(up_row0 + 3) * quant_groups + quant_group]);
        const float gs0 = bf16_to_float(gate_up_scales[(gate_row0) * quant_groups + quant_group]);
        const float gs1 = bf16_to_float(gate_up_scales[(gate_row0 + 1) * quant_groups + quant_group]);
        const float gs2 = bf16_to_float(gate_up_scales[(gate_row0 + 2) * quant_groups + quant_group]);
        const float gs3 = bf16_to_float(gate_up_scales[(gate_row0 + 3) * quant_groups + quant_group]);
        const float gb0 = bf16_to_float(gate_up_biases[(gate_row0) * quant_groups + quant_group]);
        const float gb1 = bf16_to_float(gate_up_biases[(gate_row0 + 1) * quant_groups + quant_group]);
        const float gb2 = bf16_to_float(gate_up_biases[(gate_row0 + 2) * quant_groups + quant_group]);
        const float gb3 = bf16_to_float(gate_up_biases[(gate_row0 + 3) * quant_groups + quant_group]);
        up0 += dot_q8(up_weights0, x_tile, reduction, packed_offset, us0, ub0);
        up1 += dot_q8(up_weights1, x_tile, reduction, packed_offset, us1, ub1);
        up2 += dot_q8(up_weights2, x_tile, reduction, packed_offset, us2, ub2);
        up3 += dot_q8(up_weights3, x_tile, reduction, packed_offset, us3, ub3);
        gate0 += dot_q8(gate_weights0, x_tile, reduction, packed_offset, gs0, gb0);
        gate1 += dot_q8(gate_weights1, x_tile, reduction, packed_offset, gs1, gb1);
        gate2 += dot_q8(gate_weights2, x_tile, reduction, packed_offset, gs2, gb2);
        gate3 += dot_q8(gate_weights3, x_tile, reduction, packed_offset, gs3, gb3);
      }
    }

    up0 = simd_sum(up0);
    up1 = simd_sum(up1);
    up2 = simd_sum(up2);
    up3 = simd_sum(up3);
    gate0 = simd_sum(gate0);
    gate1 = simd_sum(gate1);
    gate2 = simd_sum(gate2);
    gate3 = simd_sum(gate3);
    if (lane_id == 0) {
      const uint output_base = expert_slot * 512 + channel_base + channel_block;
      intermediate[output_base] =
          (gate0 / (1.0f + metal::exp(-gate0))) * up0;
      intermediate[output_base + 1] =
          (gate1 / (1.0f + metal::exp(-gate1))) * up1;
      intermediate[output_base + 2] =
          (gate2 / (1.0f + metal::exp(-gate2))) * up2;
      intermediate[output_base + 3] =
          (gate3 / (1.0f + metal::exp(-gate3))) * up3;
    }
  }
}

kernel void down_router(
    const device uint* down_weights [[buffer(0)]],
    const device ushort* down_scales [[buffer(1)]],
    const device ushort* down_biases [[buffer(2)]],
    const device float* intermediate [[buffer(3)]],
    const device uint* selected_experts [[buffer(4)]],
    const device float* router_weights [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant FusedMoeParams& params [[buffer(7)]],
    const uint group_id [[threadgroup_position_in_grid]],
    const uint thread_id [[thread_index_in_threadgroup]],
    const uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    const uint lane_id [[thread_index_in_simdgroup]],
    const uint threadgroup_size [[threads_per_threadgroup]]) {
  if (params.bits != 4 && params.bits != 8) {
    return;
  }

  threadgroup float x_tile[512];
  const uint simdgroups_per_threadgroup = threadgroup_size / 32;
  const uint global_simdgroup =
      group_id * simdgroups_per_threadgroup + simdgroup_id;
  const uint active_output_size = params.expert_count * params.hidden_size;
  const uint rows_per_simdgroup =
      active_output_size / (params.group_count * simdgroups_per_threadgroup);
  const uint first_active_row = global_simdgroup * rows_per_simdgroup;
  const uint expert_slot = first_active_row / params.hidden_size;
  const uint expert_id = selected_experts[expert_slot];
  const uint first_weight_row = expert_id * params.hidden_size;
  const uint reduction_size = params.intermediate_size;
  const uint values_per_pack = 32 / params.bits;
  const uint packed_stride = reduction_size / values_per_pack;
  const uint quant_groups = reduction_size / 64;
  const uint reduction_start = lane_id * values_per_pack;
  const uint reduction_stride = values_per_pack * 32;

  for (uint index = thread_id; index < reduction_size; index += threadgroup_size) {
    x_tile[index] = intermediate[expert_slot * reduction_size + index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row_block = 0; row_block < rows_per_simdgroup; row_block += 4) {
    const uint active_row0 = first_active_row + row_block;
    const uint active_row1 = active_row0 + 1;
    const uint active_row2 = active_row0 + 2;
    const uint active_row3 = active_row0 + 3;
    const uint row0 = first_weight_row + row_block;
    const uint row1 = row0 + 1;
    const uint row2 = row0 + 2;
    const uint row3 = row0 + 3;
    const device uint* weight0 = down_weights + row0 * packed_stride;
    const device uint* weight1 = down_weights + row1 * packed_stride;
    const device uint* weight2 = down_weights + row2 * packed_stride;
    const device uint* weight3 = down_weights + row3 * packed_stride;
    float value0 = 0.0f;
    float value1 = 0.0f;
    float value2 = 0.0f;
    float value3 = 0.0f;

    for (uint reduction = reduction_start; reduction < reduction_size;
         reduction += reduction_stride) {
      const uint packed_offset = reduction / values_per_pack;
      const uint quant_group = reduction / 64;
      if (params.bits == 4) {
        const float scale0 = bf16_to_float(down_scales[row0 * quant_groups + quant_group]);
        const float scale1 = bf16_to_float(down_scales[row1 * quant_groups + quant_group]);
        const float scale2 = bf16_to_float(down_scales[row2 * quant_groups + quant_group]);
        const float scale3 = bf16_to_float(down_scales[row3 * quant_groups + quant_group]);
        const float bias0 = bf16_to_float(down_biases[row0 * quant_groups + quant_group]);
        const float bias1 = bf16_to_float(down_biases[row1 * quant_groups + quant_group]);
        const float bias2 = bf16_to_float(down_biases[row2 * quant_groups + quant_group]);
        const float bias3 = bf16_to_float(down_biases[row3 * quant_groups + quant_group]);
        value0 += dot_q4(weight0, x_tile, reduction, packed_offset, scale0, bias0);
        value1 += dot_q4(weight1, x_tile, reduction, packed_offset, scale1, bias1);
        value2 += dot_q4(weight2, x_tile, reduction, packed_offset, scale2, bias2);
        value3 += dot_q4(weight3, x_tile, reduction, packed_offset, scale3, bias3);
      } else {
        const float scale0 = bf16_to_float(down_scales[row0 * quant_groups + quant_group]);
        const float scale1 = bf16_to_float(down_scales[row1 * quant_groups + quant_group]);
        const float scale2 = bf16_to_float(down_scales[row2 * quant_groups + quant_group]);
        const float scale3 = bf16_to_float(down_scales[row3 * quant_groups + quant_group]);
        const float bias0 = bf16_to_float(down_biases[row0 * quant_groups + quant_group]);
        const float bias1 = bf16_to_float(down_biases[row1 * quant_groups + quant_group]);
        const float bias2 = bf16_to_float(down_biases[row2 * quant_groups + quant_group]);
        const float bias3 = bf16_to_float(down_biases[row3 * quant_groups + quant_group]);
        value0 += dot_q8(weight0, x_tile, reduction, packed_offset, scale0, bias0);
        value1 += dot_q8(weight1, x_tile, reduction, packed_offset, scale1, bias1);
        value2 += dot_q8(weight2, x_tile, reduction, packed_offset, scale2, bias2);
        value3 += dot_q8(weight3, x_tile, reduction, packed_offset, scale3, bias3);
      }
    }

    value0 = simd_sum(value0);
    value1 = simd_sum(value1);
    value2 = simd_sum(value2);
    value3 = simd_sum(value3);
    if (lane_id == 0) {
      output[active_row0] = value0;
      output[active_row1] = value1;
      output[active_row2] = value2;
      output[active_row3] = value3;
    }
  }
}

kernel void combine_router(
    const device float* active_output [[buffer(0)]],
    const device float* router_weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant FusedMoeParams& params [[buffer(3)]],
    const uint group_id [[threadgroup_position_in_grid]],
    const uint thread_id [[thread_index_in_threadgroup]]) {
  const uint hidden = group_id * 256 + thread_id;
  if (hidden >= params.hidden_size) {
    return;
  }
  float sum = 0.0f;
  for (uint slot = 0; slot < params.expert_count; ++slot) {
    sum += router_weights[slot] *
        active_output[slot * params.hidden_size + hidden];
  }
  output[hidden] = sum;
}