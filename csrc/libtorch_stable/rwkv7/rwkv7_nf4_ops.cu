// NF4 (E2M1) quantized linear kernels for RWKV-7 v3a
// Follows the exact same structure as rwkv7_v3a_ops.cu
// Weight: uint8 [N, K/2] packed (2 E2M1 per byte)
// Block scale: __half [N, K/16] (per-block, 16 elements along K)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <vector>

using dtype = at::Half;

namespace {

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

// E2M1 4-bit float -> float32 decode table (16 entries)
// Index 0-7: positive values, Index 8-15: negative values
__constant__ float e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

// ═══════════════════════════════════════════════════════════════
// M=1 GEMV: x[K] @ w[N,K]^T -> y[N]
// 2-wide K loop (same as v3a linear_orig_row1_exact_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_row1_exact_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const dtype* __restrict__ b_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int K2 = K >> 1;    // packed bytes per row
  const int KB = K >> 4;    // blocks per row (K/16)
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint8_t packed = w_nf4[static_cast<int64_t>(n0 + j) * K2 + k2];
      const float bs = __half2float(b_scale[static_cast<int64_t>(n0 + j) * KB + (k >> 4)]);
      acc[j] = fmaf(xv.x, e2m1_lut[packed & 0x0F] * bs, acc[j]);
      acc[j] = fmaf(xv.y, e2m1_lut[packed >> 4] * bs, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// M=1 GEMV: 4-wide K loop (same as v3a linear_orig_row1_exact4_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_row1_exact4_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const dtype* __restrict__ b_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int K2 = K >> 1;    // packed bytes per row
  const int KB = K >> 4;    // blocks per row
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  // 4-wide: process 4 K elements per iteration (2 packed bytes)
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const int k2 = k >> 1;  // byte index
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint8_t* wp = w_nf4 + static_cast<int64_t>(n0 + j) * K2 + k2;
      // k, k+1, k+2, k+3 all within same 16-element block (k is 4-aligned)
      const float bs = __half2float(b_scale[static_cast<int64_t>(n0 + j) * KB + (k >> 4)]);
      const float wv0 = e2m1_lut[wp[0] & 0x0F] * bs;
      const float wv1 = e2m1_lut[wp[0] >> 4]  * bs;
      const float wv2 = e2m1_lut[wp[1] & 0x0F] * bs;
      const float wv3 = e2m1_lut[wp[1] >> 4]  * bs;
      acc[j] = fmaf(x0.x, wv0, acc[j]);
      acc[j] = fmaf(x0.y, wv1, acc[j]);
      acc[j] = fmaf(x1.x, wv2, acc[j]);
      acc[j] = fmaf(x1.y, wv3, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// M=2 GEMV: x[2,K] @ w[N,K]^T -> y[2,N]
// 2-wide K loop (same as v3a linear_orig_row2_exact_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_row2_exact_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const dtype* __restrict__ b_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int K2 = K >> 1;
  const int KB = K >> 4;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint8_t packed = w_nf4[static_cast<int64_t>(n0 + j) * K2 + k2];
      const float bs = __half2float(b_scale[static_cast<int64_t>(n0 + j) * KB + (k >> 4)]);
      const float wv_x = e2m1_lut[packed & 0x0F] * bs;
      const float wv_y = e2m1_lut[packed >> 4]  * bs;
      acc0[j] = fmaf(x0.x, wv_x, acc0[j]);
      acc0[j] = fmaf(x0.y, wv_y, acc0[j]);
      acc1[j] = fmaf(x1.x, wv_x, acc1[j]);
      acc1[j] = fmaf(x1.y, wv_y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// M=2 GEMV: 4-wide K loop (same as v3a linear_orig_row2_exact4_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_row2_exact4_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const dtype* __restrict__ b_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int K2 = K >> 1;
  const int KB = K >> 4;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x00 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x01 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const float2 x10 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
    const float2 x11 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 2));
    const int k2 = k >> 1;
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const uint8_t* wp = w_nf4 + static_cast<int64_t>(n0 + j) * K2 + k2;
      const float bs = __half2float(b_scale[static_cast<int64_t>(n0 + j) * KB + (k >> 4)]);
      const float wv0 = e2m1_lut[wp[0] & 0x0F] * bs;
      const float wv1 = e2m1_lut[wp[0] >> 4]  * bs;
      const float wv2 = e2m1_lut[wp[1] & 0x0F] * bs;
      const float wv3 = e2m1_lut[wp[1] >> 4]  * bs;
      acc0[j] = fmaf(x00.x, wv0, acc0[j]);
      acc0[j] = fmaf(x00.y, wv1, acc0[j]);
      acc0[j] = fmaf(x01.x, wv2, acc0[j]);
      acc0[j] = fmaf(x01.y, wv3, acc0[j]);
      acc1[j] = fmaf(x10.x, wv0, acc1[j]);
      acc1[j] = fmaf(x10.y, wv1, acc1[j]);
      acc1[j] = fmaf(x11.x, wv2, acc1[j]);
      acc1[j] = fmaf(x11.y, wv3, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// M>=3 GEMM: x[M,K] @ w[N,K]^T -> y[M,N]
// 2-wide K loop (same as v3a linear_orig_rows_f16_kernel)
// ═══════════════════════════════════════════════════════════════
template <int Threads, int RowTile, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_nf4_orig_rows_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const uint8_t* __restrict__ w_nf4,
    const dtype* __restrict__ b_scale,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m0 = blockIdx.y * RowTile;
  const int K2 = K >> 1;
  const int KB = K >> 4;
  float acc[RowTile][OutTile];
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      acc[r][j] = 0.0f;
    }
  }
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    // Decode weight once, share across RowTile rows (same pattern as v3a)
    float wv_low[OutTile];
    float wv_high[OutTile];
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const uint8_t packed = w_nf4[static_cast<int64_t>(n) * K2 + k2];
        const float bs = __half2float(b_scale[static_cast<int64_t>(n) * KB + (k >> 4)]);
        wv_low[j]  = e2m1_lut[packed & 0x0F] * bs;
        wv_high[j] = e2m1_lut[packed >> 4]    * bs;
      } else {
        wv_low[j]  = 0.0f;
        wv_high[j] = 0.0f;
      }
    }
    // Reuse decoded weight across all rows
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k));
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          acc[r][j] = fmaf(xv.x, wv_low[j], acc[r][j]);
          acc[r][j] = fmaf(xv.y, wv_high[j], acc[r][j]);
        }
      }
    }
  }
  __shared__ float partial[Threads / 32][RowTile][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float v = warp_sum(acc[r][j]);
      if (lane == 0) {
        partial[warp][r][j] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          const int n = n0 + j;
          if (n < N) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < Threads / 32; ++w) {
              sum += partial[w][r][j];
            }
            *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
          }
        }
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// Batch dequant: NF4 [N, K/2] uint8 + b_scale [N, K/16] -> fp16
// Transpose=false: output [N, K] (orig layout)
// Transpose=true:  output [K, N] (non-orig layout)
// ═══════════════════════════════════════════════════════════════
template <int Threads, bool Transpose>
__global__ void dequant_nf4_to_f16_kernel(
    int N,
    int K,
    const uint8_t* __restrict__ w_nf4,
    const dtype* __restrict__ b_scale,
    dtype* __restrict__ out) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const int K2 = K >> 1;
  const int KB = K >> 4;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const int k2 = k >> 1;
    const uint8_t packed = w_nf4[static_cast<int64_t>(n) * K2 + k2];
    const float bs = __half2float(b_scale[static_cast<int64_t>(n) * KB + (k >> 4)]);
    float val;
    if ((k & 1) == 0) {
      val = e2m1_lut[packed & 0x0F] * bs;
    } else {
      val = e2m1_lut[packed >> 4] * bs;
    }
    if (Transpose) {
      out[static_cast<int64_t>(k) * N + n] = __float2half_rn(val);
    } else {
      out[static_cast<int64_t>(n) * K + k] = __float2half_rn(val);
    }
  }
}

// ═══════════════════════════════════════════════════════════════
// Host wrappers
// ═══════════════════════════════════════════════════════════════

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_nf4_orig_row1_exact_f16_cuda_impl(at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_nf4.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_nf4 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_nf4 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_nf4 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 1, "linear_nf4 row1 requires one row");
  TORCH_CHECK(w_nf4.size(1) == K / 2, "linear_nf4 w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "linear_nf4 b_scale shape mismatch");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (Use4) {
    linear_nf4_orig_row1_exact4_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, stream>>>(
        K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
        w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const dtype*>(b_scale.data_ptr()),
        reinterpret_cast<dtype*>(y.data_ptr()));
  } else {
    linear_nf4_orig_row1_exact_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, stream>>>(
        K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
        w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const dtype*>(b_scale.data_ptr()),
        reinterpret_cast<dtype*>(y.data_ptr()));
  }
  return y;
}

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_nf4_orig_row2_exact_f16_cuda_impl(at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_nf4.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_nf4 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_nf4 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_nf4 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 2, "linear_nf4 row2 requires two rows");
  TORCH_CHECK(w_nf4.size(1) == K / 2, "linear_nf4 w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "linear_nf4 b_scale shape mismatch");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (Use4) {
    linear_nf4_orig_row2_exact4_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, stream>>>(
        K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
        w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const dtype*>(b_scale.data_ptr()),
        reinterpret_cast<dtype*>(y.data_ptr()));
  } else {
    linear_nf4_orig_row2_exact_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, stream>>>(
        K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
        w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const dtype*>(b_scale.data_ptr()),
        reinterpret_cast<dtype*>(y.data_ptr()));
  }
  return y;
}

template <int Threads, int RowTile, int OutTile>
at::Tensor linear_nf4_orig_rows_f16_cuda_impl(at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = w_nf4.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_nf4 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  const int M = static_cast<int>(m64);
  TORCH_CHECK(w_nf4.size(1) == K / 2, "linear_nf4 w_nf4 K/2 mismatch");
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "linear_nf4 b_scale shape mismatch");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(ceil_div(N, OutTile), ceil_div(M, RowTile), 1);
  linear_nf4_orig_rows_f16_kernel<Threads, RowTile, OutTile><<<grid, Threads, 0, stream>>>(
      M, K, N, reinterpret_cast<const dtype*>(x.data_ptr()),
      w_nf4.data_ptr<uint8_t>(), reinterpret_cast<const dtype*>(b_scale.data_ptr()),
      reinterpret_cast<dtype*>(y.data_ptr()));
  return y;
}

} // namespace

// ═══════════════════════════════════════════════════════════════
// C++ entry points (called from .cpp)
// ═══════════════════════════════════════════════════════════════

at::Tensor linear_nf4_orig_rows_exact_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, int64_t threads, int64_t out_tile, bool use4) {
  // Dispatch by rows (same as v3a: x.numel() / x.size(-1))
  const int64_t rows = x.numel() / x.size(-1);
  if (rows == 1) {
    if (!use4 && threads == 128 && out_tile == 2) return linear_nf4_orig_row1_exact_f16_cuda_impl<128, 2, false>(x, w_nf4, b_scale);
    if (use4 && threads == 128 && out_tile == 2) return linear_nf4_orig_row1_exact_f16_cuda_impl<128, 2, true>(x, w_nf4, b_scale);
  }
  if (rows == 2) {
    if (use4 && threads == 64 && out_tile == 2) return linear_nf4_orig_row2_exact_f16_cuda_impl<64, 2, true>(x, w_nf4, b_scale);
    if (use4 && threads == 256 && out_tile == 1) return linear_nf4_orig_row2_exact_f16_cuda_impl<256, 1, true>(x, w_nf4, b_scale);
    if (!use4 && threads == 128 && out_tile == 2) return linear_nf4_orig_row2_exact_f16_cuda_impl<128, 2, false>(x, w_nf4, b_scale);
  }
  TORCH_CHECK(false, "unsupported linear_nf4_orig_rows_exact_f16 rows/threads/out_tile/use4");
}

at::Tensor linear_nf4_orig_rows_f16_cuda(
    at::Tensor x, at::Tensor w_nf4, at::Tensor b_scale, int64_t row_tile, int64_t out_tile) {
  // Dispatch by row_tile/out_tile (same options as v3a)
  if (row_tile == 1 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 1, 2>(x, w_nf4, b_scale);
  if (row_tile == 1 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 1, 4>(x, w_nf4, b_scale);
  if (row_tile == 1 && out_tile == 8) return linear_nf4_orig_rows_f16_cuda_impl<128, 1, 8>(x, w_nf4, b_scale);
  if (row_tile == 1 && out_tile == 16) return linear_nf4_orig_rows_f16_cuda_impl<128, 1, 16>(x, w_nf4, b_scale);
  if (row_tile == 2 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 2, 2>(x, w_nf4, b_scale);
  if (row_tile == 2 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 2, 4>(x, w_nf4, b_scale);
  if (row_tile == 2 && out_tile == 8) return linear_nf4_orig_rows_f16_cuda_impl<128, 2, 8>(x, w_nf4, b_scale);
  if (row_tile == 3 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 3, 2>(x, w_nf4, b_scale);
  if (row_tile == 3 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 3, 4>(x, w_nf4, b_scale);
  if (row_tile == 4 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 4, 2>(x, w_nf4, b_scale);
  if (row_tile == 4 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 4, 4>(x, w_nf4, b_scale);
  if (row_tile == 8 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 8, 2>(x, w_nf4, b_scale);
  if (row_tile == 8 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 8, 4>(x, w_nf4, b_scale);
  if (row_tile == 16 && out_tile == 2) return linear_nf4_orig_rows_f16_cuda_impl<128, 16, 2>(x, w_nf4, b_scale);
  if (row_tile == 16 && out_tile == 4) return linear_nf4_orig_rows_f16_cuda_impl<128, 16, 4>(x, w_nf4, b_scale);
  TORCH_CHECK(false, "unsupported linear_nf4_orig_rows_f16 row_tile/out_tile");
}

at::Tensor dequant_nf4_to_f16_cuda(at::Tensor w_nf4, at::Tensor b_scale, bool transpose) {
  const int N = static_cast<int>(w_nf4.size(0));
  const int K = static_cast<int>(w_nf4.size(1)) * 2;
  TORCH_CHECK(b_scale.size(0) == N && b_scale.size(1) == K / 16, "b_scale shape mismatch");
  auto out = at::empty(transpose ? std::vector<int64_t>{K, N} : std::vector<int64_t>{N, K}, w_nf4.options().dtype(at::kHalf));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  if (transpose) {
    dequant_nf4_to_f16_kernel<256, true><<<N, threads, 0, stream>>>(
        N, K, w_nf4.data_ptr<uint8_t>(),
        reinterpret_cast<const dtype*>(b_scale.data_ptr()),
        reinterpret_cast<dtype*>(out.data_ptr()));
  } else {
    dequant_nf4_to_f16_kernel<256, false><<<N, threads, 0, stream>>>(
        N, K, w_nf4.data_ptr<uint8_t>(),
        reinterpret_cast<const dtype*>(b_scale.data_ptr()),
        reinterpret_cast<dtype*>(out.data_ptr()));
  }
  return out;
}
