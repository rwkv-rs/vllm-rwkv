// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from BlinkDL/Albatross faster3a_2605/cuda at commit
// 5e941fb1eeb7f735a562fb5bbb30fad19adc825b and normalized to the same single
// packed-varlen, slot-mapped contract as the FP16-state implementation.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {

constexpr int HEAD_SIZE = 64;
constexpr float W_SCALE_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;

#ifdef _IO_FP16_
using io_t = __half;
__device__ __forceinline__ float io_to_float(io_t value) {
  return __half2float(value);
}
__device__ __forceinline__ io_t float_to_io(float value) {
  return __float2half_rn(value);
}
#else
using io_t = float;
__device__ __forceinline__ float io_to_float(float value) { return value; }
__device__ __forceinline__ float float_to_io(float value) { return value; }
#endif

__device__ __forceinline__ float load_io(const io_t* ptr, int64_t index) {
  return io_to_float(__ldg(ptr + index));
}

__device__ __forceinline__ float w_eff(float w) {
  return exp2f(W_SCALE_LOG2_E / (1.0f + exp2f(NLOG2_E * w)));
}

__global__ __launch_bounds__(HEAD_SIZE, 2) void wkv_fp32_kernel(
    int C, int H, const int* __restrict__ query_start_loc,
    const int* __restrict__ slot_indices, float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr, const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ k_ptr, const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr, const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int h = static_cast<int>(blockIdx.x);
  const int request = static_cast<int>(blockIdx.y);
  const int row = static_cast<int>(threadIdx.x);
  const int lane = row & 31;

  int token_start = 0;
  int token_end = 0;
  int state_slot = 0;
  if (lane == 0) {
    token_start = query_start_loc[request];
    token_end = query_start_loc[request + 1];
    state_slot = slot_indices[request];
  }
  token_start = __shfl_sync(0xffffffffu, token_start, 0);
  token_end = __shfl_sync(0xffffffffu, token_end, 0);
  state_slot = __shfl_sync(0xffffffffu, state_slot, 0);
  const int token_count = token_end - token_start;
  if (token_count <= 0) {
    return;
  }

  float* state_base = state_ptr + (static_cast<int64_t>(state_slot) * H *
                                       HEAD_SIZE * HEAD_SIZE +
                                   h * HEAD_SIZE * HEAD_SIZE + row * HEAD_SIZE);
  float state[HEAD_SIZE];
#pragma unroll
  for (int column = 0; column < HEAD_SIZE; ++column) {
    state[column] = state_base[column];
  }

  __shared__ float r[HEAD_SIZE];
  __shared__ float w[HEAD_SIZE];
  __shared__ float k[HEAD_SIZE];
  __shared__ float a[HEAD_SIZE];
  __shared__ float b[HEAD_SIZE];

  const int channel_base = h * HEAD_SIZE;
  for (int offset = 0; offset < token_count; ++offset) {
    const int64_t index =
        static_cast<int64_t>(token_start + offset) * C + channel_base + row;
    __syncthreads();
    r[row] = load_io(r_ptr, index);
    w[row] = w_eff(load_io(w_ptr, index));
    k[row] = load_io(k_ptr, index);
    a[row] = load_io(a_ptr, index);
    b[row] = load_io(b_ptr, index);
    __syncthreads();

    float state_dot_a = 0.0f;
#pragma unroll
    for (int column = 0; column < HEAD_SIZE; ++column) {
      state_dot_a += state[column] * a[column];
    }

    const float value = load_io(v_ptr, index);
    float output = 0.0f;
#pragma unroll
    for (int column = 0; column < HEAD_SIZE; ++column) {
      float recurrent = state[column];
      recurrent =
          recurrent * w[column] + state_dot_a * b[column] + k[column] * value;
      state[column] = recurrent;
      output += recurrent * r[column];
    }
    y_ptr[index] = float_to_io(output);
  }

#pragma unroll
  for (int column = 0; column < HEAD_SIZE; ++column) {
    state_base[column] = state[column];
  }
}

}  // namespace

void wkv_fp32_cuda(int B, int C, int H, at::Tensor query_start_loc,
                   at::Tensor slot_indices, at::Tensor state, at::Tensor r,
                   at::Tensor w, at::Tensor k, at::Tensor v, at::Tensor a,
                   at::Tensor b, at::Tensor y) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp32_kernel<<<dim3(H, B), dim3(HEAD_SIZE), 0, stream>>>(
      C, H, query_start_loc.data_ptr<int>(), slot_indices.data_ptr<int>(),
      state.data_ptr<float>(), reinterpret_cast<const io_t*>(r.data_ptr()),
      reinterpret_cast<const io_t*>(w.data_ptr()),
      reinterpret_cast<const io_t*>(k.data_ptr()),
      reinterpret_cast<const io_t*>(v.data_ptr()),
      reinterpret_cast<const io_t*>(a.data_ptr()),
      reinterpret_cast<const io_t*>(b.data_ptr()),
      reinterpret_cast<io_t*>(y.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
