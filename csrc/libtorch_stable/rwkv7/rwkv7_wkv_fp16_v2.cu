// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// The recurrent state layout, half2 arithmetic, swizzled state staging, and
// cp.async pipeline are adapted from BlinkDL/Albatross faster3a_2607/cuda at
// commit 63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b. This file extends that
// implementation with packed-varlen request metadata and slot-mapped state.
// Source:
// https://github.com/BlinkDL/Albatross/tree/63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b/faster3a_2607/cuda
// Upstream license: Apache-2.0
// (https://github.com/BlinkDL/Albatross/blob/63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b/LICENSE).

#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr int HEAD_SIZE = 64;
constexpr int HALF2_HEAD_SIZE = HEAD_SIZE / 2;
constexpr int LDG_ELEMS = sizeof(int4) / sizeof(half);
constexpr float TWO_NEG_41 = 4.547473508864641e-13f;
constexpr float NEXP_HALF_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;
constexpr uint32_t ROT1 = 2654435769u;

__device__ __forceinline__ float rotator1(int x) {
  const uint32_t bits = ROT1 * static_cast<uint32_t>(x);
  return TWO_NEG_41 * static_cast<float>(static_cast<int32_t>(bits));
}

__device__ __forceinline__ half w_delta(half w_raw, half w0, int phase) {
  const float w = __half2float(w_raw) + __half2float(w0);
  const float delta = exp2f(NEXP_HALF_LOG2_E / (1.0f + exp2f(NLOG2_E * w))) -
                      1.0f + rotator1(phase);
  return __float2half_rn(delta);
}

template <int Bytes>
__device__ __forceinline__ void cp_async(void* smem, const void* global,
                                         bool pred) {
  static_assert(Bytes == 16 || Bytes == 8 || Bytes == 4);
  const int bytes = pred ? Bytes : 0;
  const unsigned int addr = __cvta_generic_to_shared(smem);
  if constexpr (Bytes == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;" ::"r"(addr),
                 "l"(global), "n"(Bytes), "r"(bytes));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(addr),
                 "l"(global), "n"(Bytes), "r"(bytes));
  }
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int NWait>
__device__ __forceinline__ void cp_wait() {
  if constexpr (NWait == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(NWait));
  }
}

__device__ __forceinline__ void prefetch_token(
    int thread, int lane, int token, half2* r, half2* w, half2* k, half2* a,
    half2* b, half2* b_dummy, const half* r_ptr, const half* w_ptr,
    const half* k_ptr, const half* a_ptr, const half* b_ptr) {
  cp_async<4>((thread < 32 ? w : a) + lane,
              reinterpret_cast<const half2*>(thread < 32 ? w_ptr + token
                                                         : a_ptr + token) +
                  lane,
              true);
  cp_commit();
  cp_async<4>((thread < 32 ? r : k) + lane,
              reinterpret_cast<const half2*>(thread < 32 ? r_ptr + token
                                                         : k_ptr + token) +
                  lane,
              true);
  // A predicated-off cp.async writes zero. The second warp therefore needs a
  // distinct sink instead of racing the first warp's b vector.
  cp_async<4>((thread < 32 ? b : b_dummy) + lane,
              reinterpret_cast<const half2*>(b_ptr + token) + lane,
              thread < 32);
  cp_commit();
}

__global__ __launch_bounds__(HEAD_SIZE, 2) void wkv_fp16_kernel(
    int C, const int* __restrict__ query_start_loc,
    const int* __restrict__ slot_indices, half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr, const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr, const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr, const half* __restrict__ a_ptr,
    const half* __restrict__ b_ptr, half* __restrict__ y_ptr,
    const int* __restrict__ elapsed_t) {
  const int h = static_cast<int>(blockIdx.x);
  const int request = static_cast<int>(blockIdx.y);
  const int thread = static_cast<int>(threadIdx.x);
  const int lane = thread & 31;

  // Each warp reads the three request scalars once and broadcasts them. This
  // avoids both a block-wide metadata barrier and 64 redundant global loads.
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

  __shared__ __align__(256) half2 state_smem[HEAD_SIZE][HALF2_HEAD_SIZE];
  state_ptr += static_cast<int64_t>(state_slot) * C * HEAD_SIZE +
               h * HEAD_SIZE * HEAD_SIZE;

#pragma unroll
  for (int j0 = 0; j0 < HEAD_SIZE / LDG_ELEMS; ++j0) {
    const int4 state_vec =
        reinterpret_cast<const int4*>(state_ptr)[j0 * HEAD_SIZE + thread];
#pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      const int row = j0 * LDG_ELEMS + thread * LDG_ELEMS / HEAD_SIZE;
      const int col = thread * LDG_ELEMS % HEAD_SIZE / 2 + j1;
      state_smem[row][(row & 31) ^ col] =
          reinterpret_cast<const half2*>(&state_vec)[j1];
    }
  }
  __syncthreads();

  half2 state[HALF2_HEAD_SIZE];
#pragma unroll
  for (int j = 0; j < HALF2_HEAD_SIZE; ++j) {
    state[j] = state_smem[thread][lane ^ j];
  }

  __shared__ __align__(128) half2 r[2][HALF2_HEAD_SIZE], w[2][HALF2_HEAD_SIZE],
      k[2][HALF2_HEAD_SIZE], a[2][HALF2_HEAD_SIZE], bvec[2][HALF2_HEAD_SIZE],
      bvec_dummy[HALF2_HEAD_SIZE];

  const int channel = h * HEAD_SIZE + thread;
  const half w0 = w0_ptr[channel];
  const int phase0 = elapsed_t[state_slot] + channel;
  int token = token_start * C + h * HEAD_SIZE;
  prefetch_token(thread, lane, token, r[0], w[0], k[0], a[0], bvec[0],
                 bvec_dummy, r_ptr, w_ptr, k_ptr, a_ptr, b_ptr);

  for (int offset = 0; offset < token_count; ++offset) {
    const int current = offset & 1;
    const half value = v_ptr[token + thread];
    const half2 value2 = {value, value};

    // Wait only for w/a first. Their dot product and decay transform overlap
    // the second r/k/b copy group, matching the short-sequence exact pipeline.
    cp_wait<1>();
    __syncthreads();

    half2 state_dot_a2 = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < HALF2_HEAD_SIZE; ++j) {
      state_dot_a2 = __hfma2(a[current][j], state[j], state_dot_a2);
    }
    const half state_dot_a = state_dot_a2.x + state_dot_a2.y;
    state_dot_a2 = {state_dot_a, state_dot_a};
    reinterpret_cast<half*>(w[current])[thread] = w_delta(
        reinterpret_cast<half*>(w[current])[thread], w0, phase0 + offset);

    cp_wait<0>();
    __syncthreads();

    // Start the next token before the recurrent update. The alternate shared
    // buffer makes this overlap safe and preserves the long-sequence pipeline.
    if (offset + 1 < token_count) {
      const int next_token = token + C;
      prefetch_token(thread, lane, next_token, r[current ^ 1], w[current ^ 1],
                     k[current ^ 1], a[current ^ 1], bvec[current ^ 1],
                     bvec_dummy, r_ptr, w_ptr, k_ptr, a_ptr, b_ptr);
    }

    half2 output2 = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < HALF2_HEAD_SIZE; ++j) {
      half2 recurrent = state[j];
      recurrent =
          __hfma2(recurrent, w[current][j],
                  __hfma2(k[current][j], value2,
                          __hfma2(state_dot_a2, bvec[current][j], recurrent)));
      state[j] = recurrent;
      output2 = __hfma2(recurrent, r[current][j], output2);
    }
    y_ptr[token + thread] = output2.x + output2.y;
    token += C;
  }

#pragma unroll
  for (int j = 0; j < HALF2_HEAD_SIZE; ++j) {
    state_smem[thread][lane ^ j] = state[j];
  }
  __syncthreads();
#pragma unroll
  for (int j0 = 0; j0 < HEAD_SIZE / LDG_ELEMS; ++j0) {
    int4 state_vec;
#pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      const int row = j0 * LDG_ELEMS + thread * LDG_ELEMS / HEAD_SIZE;
      const int col = thread * LDG_ELEMS % HEAD_SIZE / 2 + j1;
      reinterpret_cast<half2*>(&state_vec)[j1] =
          state_smem[row][(row & 31) ^ col];
    }
    reinterpret_cast<int4*>(state_ptr)[j0 * HEAD_SIZE + thread] = state_vec;
  }
}

}  // namespace

void wkv_fp16_cuda(int B, int C, int H, at::Tensor query_start_loc,
                   at::Tensor slot_indices, at::Tensor state, at::Tensor r,
                   at::Tensor w, at::Tensor w0, at::Tensor k, at::Tensor v,
                   at::Tensor a, at::Tensor b, at::Tensor y,
                   at::Tensor elapsed_t) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp16_kernel<<<dim3(H, B), dim3(HEAD_SIZE), 0, stream>>>(
      C, query_start_loc.data_ptr<int>(), slot_indices.data_ptr<int>(),
      reinterpret_cast<half*>(state.data_ptr()),
      reinterpret_cast<const half*>(r.data_ptr()),
      reinterpret_cast<const half*>(w.data_ptr()),
      reinterpret_cast<const half*>(w0.data_ptr()),
      reinterpret_cast<const half*>(k.data_ptr()),
      reinterpret_cast<const half*>(v.data_ptr()),
      reinterpret_cast<const half*>(a.data_ptr()),
      reinterpret_cast<const half*>(b.data_ptr()),
      reinterpret_cast<half*>(y.data_ptr()), elapsed_t.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
