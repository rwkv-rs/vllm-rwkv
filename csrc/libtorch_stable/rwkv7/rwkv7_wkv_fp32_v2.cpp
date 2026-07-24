// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <torch/library.h>

#include <limits>
#include <utility>

void wkv_fp32_cuda(int B, int C, int H, torch::Tensor query_start_loc,
                   torch::Tensor slot_indices, torch::Tensor state,
                   torch::Tensor r, torch::Tensor w, torch::Tensor k,
                   torch::Tensor v, torch::Tensor a, torch::Tensor b,
                   torch::Tensor y);

namespace {

constexpr int64_t HEAD_SIZE = 64;

#ifdef _IO_FP16_
constexpr auto IO_DTYPE = torch::kFloat16;
constexpr const char* IO_DTYPE_NAME = "fp16";
#else
constexpr auto IO_DTYPE = torch::kFloat32;
constexpr const char* IO_DTYPE_NAME = "fp32";
#endif

void check_cuda_contiguous(const torch::Tensor& tensor, torch::ScalarType dtype,
                           const char* dtype_name, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " must be ", dtype_name);
}

void check_same_device(const torch::Tensor& reference,
                       const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device() == reference.device(), name,
              " must be on the same device as state");
}

}  // namespace

void wkv(torch::Tensor query_start_loc, torch::Tensor slot_indices,
         torch::Tensor state, torch::Tensor r, torch::Tensor w, torch::Tensor k,
         torch::Tensor v, torch::Tensor a, torch::Tensor b, torch::Tensor y) {
  check_cuda_contiguous(query_start_loc, torch::kInt32, "int32",
                        "query_start_loc");
  check_cuda_contiguous(slot_indices, torch::kInt32, "int32", "slot_indices");
  check_cuda_contiguous(state, torch::kFloat32, "fp32", "state");
  check_cuda_contiguous(r, IO_DTYPE, IO_DTYPE_NAME, "r");
  check_cuda_contiguous(w, IO_DTYPE, IO_DTYPE_NAME, "w");
  check_cuda_contiguous(k, IO_DTYPE, IO_DTYPE_NAME, "k");
  check_cuda_contiguous(v, IO_DTYPE, IO_DTYPE_NAME, "v");
  check_cuda_contiguous(a, IO_DTYPE, IO_DTYPE_NAME, "a");
  check_cuda_contiguous(b, IO_DTYPE, IO_DTYPE_NAME, "b");
  check_cuda_contiguous(y, IO_DTYPE, IO_DTYPE_NAME, "y");

  const int64_t batch_size = slot_indices.numel();
  TORCH_CHECK(batch_size > 0 && batch_size <= 65535,
              "slot_indices must contain 1..65535 requests");
  TORCH_CHECK(slot_indices.dim() == 1, "slot_indices must have shape [B]");
  TORCH_CHECK(
      query_start_loc.dim() == 1 && query_start_loc.size(0) == batch_size + 1,
      "query_start_loc must have shape [B+1]");
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 0 &&
                  state.size(2) == HEAD_SIZE && state.size(3) == HEAD_SIZE,
              "state must have shape [slots,H,64,64]");
  const int64_t head_count = state.size(1);
  const int64_t hidden_size = head_count * HEAD_SIZE;
  TORCH_CHECK(head_count > 0 && head_count <= std::numeric_limits<int>::max(),
              "head count must be positive int32");
  TORCH_CHECK(hidden_size <= std::numeric_limits<int>::max(),
              "hidden size must fit in int32");
  TORCH_CHECK(r.dim() == 2 && r.size(0) > 0 && r.size(1) == hidden_size,
              "r must have shape [total_tokens,C]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
                  r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
                  r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(r.size(0) <= std::numeric_limits<int>::max() / hidden_size,
              "packed token indexing exceeds signed int32");

  for (const auto& item :
       {std::pair<const torch::Tensor*, const char*>(&query_start_loc,
                                                     "query_start_loc"),
        std::pair<const torch::Tensor*, const char*>(&slot_indices,
                                                     "slot_indices"),
        std::pair<const torch::Tensor*, const char*>(&r, "r"),
        std::pair<const torch::Tensor*, const char*>(&w, "w"),
        std::pair<const torch::Tensor*, const char*>(&k, "k"),
        std::pair<const torch::Tensor*, const char*>(&v, "v"),
        std::pair<const torch::Tensor*, const char*>(&a, "a"),
        std::pair<const torch::Tensor*, const char*>(&b, "b"),
        std::pair<const torch::Tensor*, const char*>(&y, "y")}) {
    check_same_device(state, *item.first, item.second);
  }

  wkv_fp32_cuda(static_cast<int>(batch_size), static_cast<int>(hidden_size),
                static_cast<int>(head_count), query_start_loc, slot_indices,
                state, r, w, k, v, a, b, y);
}

TORCH_LIBRARY(rwkv7_wkv_fp32_v2, m) {
  m.def(
      "wkv(Tensor query_start_loc, Tensor slot_indices, Tensor(a!) state, "
      "Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, "
      "Tensor(b!) y) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_wkv_fp32_v2, CUDA, m) { m.impl("wkv", &wkv); }
