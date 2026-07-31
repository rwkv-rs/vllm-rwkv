// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from BlinkDL/Albatross faster3a_2605/cuda at commit
// 5e941fb1eeb7f735a562fb5bbb30fad19adc825b. Source:
// https://github.com/BlinkDL/Albatross/tree/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/faster3a_2605/cuda
// Upstream license: Apache-2.0
// (https://github.com/BlinkDL/Albatross/blob/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/LICENSE).
// Dense 3D mix and 2D add-vector entrypoints follow faster3a_2607 at commit
// 63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b:
// https://github.com/BlinkDL/Albatross/tree/63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b/faster3a_2607/cuda

#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <vector>

std::vector<torch::Tensor> tmix_mix6_cuda(int B, int T, int C, torch::Tensor x,
                                          torch::Tensor shift_state,
                                          torch::Tensor x_r, torch::Tensor x_w,
                                          torch::Tensor x_k, torch::Tensor x_v,
                                          torch::Tensor x_a, torch::Tensor x_g);
std::vector<torch::Tensor> tmix_mix6_3d_cuda(
    int B, int T, int C, torch::Tensor x, torch::Tensor shift_state,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v,
    torch::Tensor x_a, torch::Tensor x_g);
std::vector<torch::Tensor> tmix_mix6_slot_cuda(
    int B, int T, int C, torch::Tensor x, torch::Tensor shift_state,
    torch::Tensor slot_indices, torch::Tensor x_r, torch::Tensor x_w,
    torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g);
std::vector<torch::Tensor> tmix_mix6_varlen_cuda(
    int B, int total_tokens, int C, torch::Tensor x, torch::Tensor shift_state,
    torch::Tensor slot_indices, torch::Tensor x_r, torch::Tensor x_w,
    torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g,
    torch::Tensor query_start_loc, torch::Tensor req_id);
std::vector<torch::Tensor> tmix_kk_a_gate_cuda(
    int B, int T, int C, int H, torch::Tensor k, torch::Tensor k_k,
    torch::Tensor a0, torch::Tensor a12, torch::Tensor k_a);
std::vector<torch::Tensor> tmix_kk_a_gate_2d_cuda(
    int B, int T, int C, int H, torch::Tensor k, torch::Tensor k_k,
    torch::Tensor a0, torch::Tensor a12, torch::Tensor k_a);

torch::Tensor tmix_lnx_rkvres_xg_cuda(int B, int T, int C, int H,
                                      torch::Tensor x, torch::Tensor r,
                                      torch::Tensor k, torch::Tensor v,
                                      torch::Tensor r_k, torch::Tensor weight,
                                      torch::Tensor bias, torch::Tensor g);
torch::Tensor tmix_lnx_rkvres_xg_warp_cuda(int B, int T, int C, int H,
                                           torch::Tensor x, torch::Tensor r,
                                           torch::Tensor k, torch::Tensor v,
                                           torch::Tensor r_k,
                                           torch::Tensor weight,
                                           torch::Tensor bias, torch::Tensor g);

torch::Tensor tmix_vres_gate_cuda(int B, int T, int C, torch::Tensor v,
                                  torch::Tensor v_first, torch::Tensor v0,
                                  torch::Tensor v12);

torch::Tensor cmix_sparse_down_relu_one_cuda(int C, int F, torch::Tensor preact,
                                             torch::Tensor value_fc);
void cmix_sparse_down_relu_one_out_cuda(int C, int F, torch::Tensor preact,
                                        torch::Tensor value_fc,
                                        torch::Tensor out);

torch::Tensor cmix_sparse_down_relu_rows_cuda(int B, int T, int C, int F,
                                              torch::Tensor preact,
                                              torch::Tensor value_fc);

torch::Tensor cmix_sparse_down_relu_rows_t512_cuda(int B, int T, int C, int F,
                                                   torch::Tensor preact,
                                                   torch::Tensor value_fc);

torch::Tensor cmix_mix_cuda(int B, int T, int C, torch::Tensor x,
                            torch::Tensor shift_state, torch::Tensor x_k);
torch::Tensor cmix_mix_3d_cuda(int B, int T, int C, torch::Tensor x,
                               torch::Tensor shift_state, torch::Tensor x_k);
torch::Tensor cmix_mix_slot_cuda(int B, int T, int C, torch::Tensor x,
                                 torch::Tensor shift_state,
                                 torch::Tensor slot_indices, torch::Tensor x_k);
torch::Tensor cmix_mix_varlen_cuda(int B, int total_tokens, int C,
                                   torch::Tensor x, torch::Tensor shift_state,
                                   torch::Tensor slot_indices,
                                   torch::Tensor x_k,
                                   torch::Tensor query_start_loc,
                                   torch::Tensor req_id);

torch::Tensor relu_square_cuda(torch::Tensor x);

torch::Tensor act_tanh_cuda(torch::Tensor x);

torch::Tensor act_sigmoid_cuda(torch::Tensor x);

torch::Tensor add_vec_cuda(int C, torch::Tensor x, torch::Tensor vec);
torch::Tensor add_vec_2d_cuda(int C, torch::Tensor x, torch::Tensor vec);

namespace {

constexpr int64_t CUDA_PORTABLE_GRID_LIMIT = 65535;
constexpr int64_t CMIX_SPARSE_C_TILE = 256;
constexpr int64_t CMIX_SPARSE_F_TILE = 128;

int checked_int_arg(int64_t x, const char* name) {
  TORCH_CHECK(x > 0 && x <= std::numeric_limits<int>::max(), name,
              " must be positive and fit in int");
  return static_cast<int>(x);
}

void check_half_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, name, " must be fp16");
}

void check_half2_aligned(const torch::Tensor& x, const char* name) {
  const auto address = reinterpret_cast<std::uintptr_t>(x.data_ptr());
  TORCH_CHECK((address % alignof(std::uint32_t)) == 0, name,
              " must be aligned for fp16x2 access");
}

void check_same_device(const torch::Tensor& reference,
                       const torch::Tensor& other, const char* name) {
  TORCH_CHECK(other.device() == reference.device(), name,
              " must be on the same CUDA device as x");
}

void check_3d(const torch::Tensor& x, int64_t B, int64_t T, int64_t C,
              const char* name) {
  check_half_cuda_contig(x, name);
  TORCH_CHECK(x.dim() == 3, name, " must have shape [B,T,C]");
  TORCH_CHECK(x.size(0) == B && x.size(1) == T && x.size(2) == C, name,
              " shape mismatch");
}

void check_2d(const torch::Tensor& x, int64_t rows, int64_t C,
              const char* name) {
  check_half_cuda_contig(x, name);
  TORCH_CHECK(x.dim() == 2, name, " must have shape [rows,C]");
  TORCH_CHECK(x.size(0) == rows && x.size(1) == C, name, " shape mismatch");
}

void check_vec(const torch::Tensor& x, int64_t C, const char* name) {
  check_half_cuda_contig(x, name);
  TORCH_CHECK(x.dim() == 1 && x.size(0) == C, name, " must have shape [C]");
}

void check_cmix_sparse_down_relu_one_inputs(int64_t C, int64_t F,
                                            const torch::Tensor& preact,
                                            const torch::Tensor& value_fc) {
  checked_int_arg(C, "C");
  checked_int_arg(F, "F");
  TORCH_CHECK((C % CMIX_SPARSE_C_TILE) == 0, "C must be divisible by 256");
  TORCH_CHECK((F % CMIX_SPARSE_F_TILE) == 0, "F must be divisible by 128");
  TORCH_CHECK(C / CMIX_SPARSE_C_TILE <= CUDA_PORTABLE_GRID_LIMIT,
              "C / 256 exceeds the portable CUDA grid limit");
  TORCH_CHECK(F / CMIX_SPARSE_F_TILE <= CUDA_PORTABLE_GRID_LIMIT,
              "F / 128 exceeds the portable CUDA grid limit");
  check_half_cuda_contig(preact, "preact");
  TORCH_CHECK(preact.dim() == 1 && preact.size(0) == F,
              "preact must have shape [F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  check_half2_aligned(value_fc, "value_fc");
  check_same_device(preact, value_fc, "value_fc");
}

void check_i32_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kInt32, name, " must be int32");
}

void check_slot_indices(const torch::Tensor& x, int64_t B) {
  check_i32_cuda_contig(x, "slot_indices");
  TORCH_CHECK(x.dim() == 1 && x.size(0) == B,
              "slot_indices must have shape [B]");
}

void check_slot_shift_state(const torch::Tensor& x, int64_t C) {
  check_half_cuda_contig(x, "shift_state");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) == C,
              "shift_state must have shape [slots,C]");
}

void check_varlen_metadata(const torch::Tensor& query_start_loc,
                           const torch::Tensor& req_id, int64_t B,
                           int64_t total_tokens) {
  check_i32_cuda_contig(query_start_loc, "query_start_loc");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  check_i32_cuda_contig(req_id, "req_id");
  TORCH_CHECK(req_id.dim() == 1 && req_id.size(0) == total_tokens,
              "req_id must have shape [total_tokens]");
}

void check_dense_mix(const torch::Tensor& x, const torch::Tensor& shift_state,
                     const torch::Tensor& x_k, int64_t B, int64_t T,
                     int64_t C) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_k, C, "x_k");
}

void check_tmix_mix6_dense(const torch::Tensor& x,
                           const torch::Tensor& shift_state,
                           const torch::Tensor& x_r, const torch::Tensor& x_w,
                           const torch::Tensor& x_k, const torch::Tensor& x_v,
                           const torch::Tensor& x_a, const torch::Tensor& x_g,
                           int64_t B, int64_t T, int64_t C) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == B &&
                  shift_state.size(1) == C,
              "shift_state must have shape [B,C]");
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
}

void check_grid_3d(int64_t B, int64_t T, const char* op_name) {
  TORCH_CHECK(B > 0 && B <= CUDA_PORTABLE_GRID_LIMIT, op_name,
              " requires 1 <= B <= 65535");
  TORCH_CHECK(T > 0 && T <= CUDA_PORTABLE_GRID_LIMIT, op_name,
              " requires 1 <= T <= 65535");
}

void check_tmix_mix6_3d(const torch::Tensor& x,
                        const torch::Tensor& shift_state,
                        const torch::Tensor& x_r, const torch::Tensor& x_w,
                        const torch::Tensor& x_k, const torch::Tensor& x_v,
                        const torch::Tensor& x_a, const torch::Tensor& x_g,
                        int64_t B, int64_t T, int64_t C) {
  check_grid_3d(B, T, "tmix_mix6_3d");
  check_tmix_mix6_dense(x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, B, T, C);
  check_half2_aligned(x, "x");
  check_half2_aligned(shift_state, "shift_state");
  check_half2_aligned(x_r, "x_r");
  check_half2_aligned(x_w, "x_w");
  check_half2_aligned(x_k, "x_k");
  check_half2_aligned(x_v, "x_v");
  check_half2_aligned(x_a, "x_a");
  check_half2_aligned(x_g, "x_g");
  check_same_device(x, shift_state, "shift_state");
  check_same_device(x, x_r, "x_r");
  check_same_device(x, x_w, "x_w");
  check_same_device(x, x_k, "x_k");
  check_same_device(x, x_v, "x_v");
  check_same_device(x, x_a, "x_a");
  check_same_device(x, x_g, "x_g");
}

}  // namespace

std::vector<torch::Tensor> tmix_mix6(int64_t B, int64_t T, int64_t C,
                                     torch::Tensor x, torch::Tensor shift_state,
                                     torch::Tensor x_r, torch::Tensor x_w,
                                     torch::Tensor x_k, torch::Tensor x_v,
                                     torch::Tensor x_a, torch::Tensor x_g) {
  check_tmix_mix6_dense(x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, B, T, C);
  return tmix_mix6_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                        checked_int_arg(C, "C"), x, shift_state, x_r, x_w, x_k,
                        x_v, x_a, x_g);
}

std::vector<torch::Tensor> tmix_mix6_3d(int64_t B, int64_t T, int64_t C,
                                        torch::Tensor x,
                                        torch::Tensor shift_state,
                                        torch::Tensor x_r, torch::Tensor x_w,
                                        torch::Tensor x_k, torch::Tensor x_v,
                                        torch::Tensor x_a, torch::Tensor x_g) {
  check_tmix_mix6_3d(x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, B, T, C);
  return tmix_mix6_3d_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                           checked_int_arg(C, "C"), x, shift_state, x_r, x_w,
                           x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> tmix_mix6_slot(
    int64_t B, int64_t T, int64_t C, torch::Tensor x, torch::Tensor shift_state,
    torch::Tensor slot_indices, torch::Tensor x_r, torch::Tensor x_w,
    torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_slot_shift_state(shift_state, C);
  check_slot_indices(slot_indices, B);
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
  return tmix_mix6_slot_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                             checked_int_arg(C, "C"), x, shift_state,
                             slot_indices, x_r, x_w, x_k, x_v, x_a, x_g);
}

std::vector<torch::Tensor> tmix_mix6_varlen(
    int64_t B, int64_t total_tokens, int64_t C, torch::Tensor x,
    torch::Tensor shift_state, torch::Tensor slot_indices, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g, torch::Tensor query_start_loc, torch::Tensor req_id) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_2d(x, total_tokens, C, "x");
  check_slot_shift_state(shift_state, C);
  check_slot_indices(slot_indices, B);
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
  check_varlen_metadata(query_start_loc, req_id, B, total_tokens);
  return tmix_mix6_varlen_cuda(
      checked_int_arg(B, "B"), checked_int_arg(total_tokens, "total_tokens"),
      checked_int_arg(C, "C"), x, shift_state, slot_indices, x_r, x_w, x_k, x_v,
      x_a, x_g, query_start_loc, req_id);
}

std::vector<torch::Tensor> tmix_kk_a_gate(int64_t B, int64_t T, int64_t C,
                                          int64_t H, torch::Tensor k,
                                          torch::Tensor k_k, torch::Tensor a0,
                                          torch::Tensor a12,
                                          torch::Tensor k_a) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  return tmix_kk_a_gate_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                             checked_int_arg(C, "C"), checked_int_arg(H, "H"),
                             k, k_k, a0, a12, k_a);
}

std::vector<torch::Tensor> tmix_kk_a_gate_2d(
    int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor k,
    torch::Tensor k_k, torch::Tensor a0, torch::Tensor a12, torch::Tensor k_a) {
  const int b = checked_int_arg(B, "B");
  const int t = checked_int_arg(T, "T");
  const int c = checked_int_arg(C, "C");
  const int h = checked_int_arg(H, "H");
  TORCH_CHECK(C == 4096 && H == 64,
              "tmix_kk_a_gate_2d requires C=4096 and H=64");
  TORCH_CHECK(static_cast<int64_t>(b) * t <= 65535,
              "tmix_kk_a_gate_2d requires B*T <= 65535");
  check_3d(k, B, T, C, "k");
  check_vec(k_k, C, "k_k");
  check_vec(a0, C, "a0");
  check_3d(a12, B, T, C, "a12");
  check_vec(k_a, C, "k_a");
  const auto device = k.device();
  TORCH_CHECK(k_k.device() == device && a0.device() == device &&
                  a12.device() == device && k_a.device() == device,
              "all tensors must be on the same CUDA device");
  return tmix_kk_a_gate_2d_cuda(b, t, c, h, k, k_k, a0, a12, k_a);
}

torch::Tensor tmix_lnx_rkvres_xg(int64_t B, int64_t T, int64_t C, int64_t H,
                                 torch::Tensor x, torch::Tensor r,
                                 torch::Tensor k, torch::Tensor v,
                                 torch::Tensor r_k, torch::Tensor weight,
                                 torch::Tensor bias, torch::Tensor g) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_3d(x, B, T, C, "x");
  check_3d(r, B, T, C, "r");
  check_3d(k, B, T, C, "k");
  check_3d(v, B, T, C, "v");
  check_3d(g, B, T, C, "g");
  check_vec(r_k, C, "r_k");
  check_vec(weight, C, "weight");
  check_vec(bias, C, "bias");
  return tmix_lnx_rkvres_xg_cuda(
      checked_int_arg(B, "B"), checked_int_arg(T, "T"), checked_int_arg(C, "C"),
      checked_int_arg(H, "H"), x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor tmix_lnx_rkvres_xg_warp(int64_t B, int64_t T, int64_t C,
                                      int64_t H, torch::Tensor x,
                                      torch::Tensor r, torch::Tensor k,
                                      torch::Tensor v, torch::Tensor r_k,
                                      torch::Tensor weight, torch::Tensor bias,
                                      torch::Tensor g) {
  TORCH_CHECK(C == 4096 && H == 64,
              "tmix_lnx_rkvres_xg_warp requires C=4096 and H=64");
  check_3d(x, B, T, C, "x");
  check_3d(r, B, T, C, "r");
  check_3d(k, B, T, C, "k");
  check_3d(v, B, T, C, "v");
  check_3d(g, B, T, C, "g");
  check_vec(r_k, C, "r_k");
  check_vec(weight, C, "weight");
  check_vec(bias, C, "bias");
  const auto device = x.device();
  TORCH_CHECK(r.device() == device && k.device() == device &&
                  v.device() == device && r_k.device() == device &&
                  weight.device() == device && bias.device() == device &&
                  g.device() == device,
              "all tensors must be on the same CUDA device");
  return tmix_lnx_rkvres_xg_warp_cuda(
      checked_int_arg(B, "B"), checked_int_arg(T, "T"), checked_int_arg(C, "C"),
      checked_int_arg(H, "H"), x, r, k, v, r_k, weight, bias, g);
}

torch::Tensor tmix_vres_gate(int64_t B, int64_t T, int64_t C, torch::Tensor v,
                             torch::Tensor v_first, torch::Tensor v0,
                             torch::Tensor v12) {
  check_3d(v, B, T, C, "v");
  check_3d(v_first, B, T, C, "v_first");
  check_vec(v0, C, "v0");
  check_3d(v12, B, T, C, "v12");
  return tmix_vres_gate_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                             checked_int_arg(C, "C"), v, v_first, v0, v12);
}

torch::Tensor cmix_sparse_down_relu_one(int64_t C, int64_t F,
                                        torch::Tensor preact,
                                        torch::Tensor value_fc) {
  check_cmix_sparse_down_relu_one_inputs(C, F, preact, value_fc);
  return cmix_sparse_down_relu_one_cuda(
      checked_int_arg(C, "C"), checked_int_arg(F, "F"), preact, value_fc);
}

void cmix_sparse_down_relu_one_out(int64_t C, int64_t F, torch::Tensor preact,
                                   torch::Tensor value_fc, torch::Tensor out) {
  check_cmix_sparse_down_relu_one_inputs(C, F, preact, value_fc);
  check_half_cuda_contig(out, "out");
  TORCH_CHECK(out.dim() == 3 && out.size(0) == 1 && out.size(1) == 1 &&
                  out.size(2) == C,
              "out must have shape [1,1,C]");
  check_half2_aligned(out, "out");
  check_same_device(preact, out, "out");
  TORCH_CHECK(!out.is_alias_of(preact), "out must not overlap preact");
  TORCH_CHECK(!out.is_alias_of(value_fc), "out must not overlap value_fc");
  cmix_sparse_down_relu_one_out_cuda(
      checked_int_arg(C, "C"), checked_int_arg(F, "F"), preact, value_fc, out);
}

torch::Tensor cmix_sparse_down_relu_rows(int64_t B, int64_t T, int64_t C,
                                         int64_t F, torch::Tensor preact,
                                         torch::Tensor value_fc) {
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 256) == 0, "C must be divisible by 256");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_rows_cuda(
      checked_int_arg(B, "B"), checked_int_arg(T, "T"), checked_int_arg(C, "C"),
      checked_int_arg(F, "F"), preact, value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows_t512(int64_t B, int64_t T, int64_t C,
                                              int64_t F, torch::Tensor preact,
                                              torch::Tensor value_fc) {
  check_3d(preact, B, T, F, "preact");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  return cmix_sparse_down_relu_rows_t512_cuda(
      checked_int_arg(B, "B"), checked_int_arg(T, "T"), checked_int_arg(C, "C"),
      checked_int_arg(F, "F"), preact, value_fc);
}

torch::Tensor cmix_mix(int64_t B, int64_t T, int64_t C, torch::Tensor x,
                       torch::Tensor shift_state, torch::Tensor x_k) {
  check_dense_mix(x, shift_state, x_k, B, T, C);
  return cmix_mix_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                       checked_int_arg(C, "C"), x, shift_state, x_k);
}

torch::Tensor cmix_mix_3d(int64_t B, int64_t T, int64_t C, torch::Tensor x,
                          torch::Tensor shift_state, torch::Tensor x_k) {
  check_grid_3d(B, T, "cmix_mix_3d");
  check_dense_mix(x, shift_state, x_k, B, T, C);
  check_half2_aligned(x, "x");
  check_half2_aligned(shift_state, "shift_state");
  check_half2_aligned(x_k, "x_k");
  check_same_device(x, shift_state, "shift_state");
  check_same_device(x, x_k, "x_k");
  return cmix_mix_3d_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                          checked_int_arg(C, "C"), x, shift_state, x_k);
}

torch::Tensor cmix_mix_slot(int64_t B, int64_t T, int64_t C, torch::Tensor x,
                            torch::Tensor shift_state,
                            torch::Tensor slot_indices, torch::Tensor x_k) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_3d(x, B, T, C, "x");
  check_slot_shift_state(shift_state, C);
  check_slot_indices(slot_indices, B);
  check_vec(x_k, C, "x_k");
  return cmix_mix_slot_cuda(checked_int_arg(B, "B"), checked_int_arg(T, "T"),
                            checked_int_arg(C, "C"), x, shift_state,
                            slot_indices, x_k);
}

torch::Tensor cmix_mix_varlen(int64_t B, int64_t total_tokens, int64_t C,
                              torch::Tensor x, torch::Tensor shift_state,
                              torch::Tensor slot_indices, torch::Tensor x_k,
                              torch::Tensor query_start_loc,
                              torch::Tensor req_id) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_2d(x, total_tokens, C, "x");
  check_slot_shift_state(shift_state, C);
  check_slot_indices(slot_indices, B);
  check_vec(x_k, C, "x_k");
  check_varlen_metadata(query_start_loc, req_id, B, total_tokens);
  return cmix_mix_varlen_cuda(checked_int_arg(B, "B"),
                              checked_int_arg(total_tokens, "total_tokens"),
                              checked_int_arg(C, "C"), x, shift_state,
                              slot_indices, x_k, query_start_loc, req_id);
}

torch::Tensor relu_square(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return relu_square_cuda(x);
}

torch::Tensor act_tanh(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return act_tanh_cuda(x);
}

torch::Tensor act_sigmoid(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  TORCH_CHECK((x.numel() % 2) == 0, "x.numel() must be even");
  return act_sigmoid_cuda(x);
}

torch::Tensor add_vec(int64_t C, torch::Tensor x, torch::Tensor vec) {
  check_half_cuda_contig(x, "x");
  check_vec(vec, C, "vec");
  TORCH_CHECK(x.numel() > 0 && (x.numel() % 2) == 0,
              "x.numel() must be positive and even");
  TORCH_CHECK(x.size(-1) == C, "x last dim must equal C");
  return add_vec_cuda(checked_int_arg(C, "C"), x, vec);
}

torch::Tensor add_vec_2d(int64_t C, torch::Tensor x, torch::Tensor vec) {
  const int c = checked_int_arg(C, "C");
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_half_cuda_contig(x, "x");
  check_vec(vec, C, "vec");
  TORCH_CHECK(x.numel() > 0 && (x.numel() % C) == 0,
              "x must contain complete C-wide rows");
  TORCH_CHECK(x.dim() > 0 && x.size(-1) == C, "x last dim must equal C");
  const int64_t rows = x.numel() / C;
  TORCH_CHECK(rows <= CUDA_PORTABLE_GRID_LIMIT,
              "add_vec_2d requires rows <= 65535");
  check_half2_aligned(x, "x");
  check_half2_aligned(vec, "vec");
  check_same_device(x, vec, "vec");
  return add_vec_2d_cuda(c, x, vec);
}

TORCH_LIBRARY(rwkv7_fast_ops_fp16, m) {
  m.def(
      "tmix_mix6(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) "
      "-> Tensor[]");
  m.def(
      "tmix_mix6_3d(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) "
      "-> Tensor[]");
  m.def(
      "tmix_mix6_slot(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor slot_indices, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, "
      "Tensor x_a, Tensor x_g) -> Tensor[]");
  m.def(
      "tmix_mix6_varlen(int B, int total_tokens, int C, Tensor x, Tensor(a!) "
      "shift_state, Tensor slot_indices, Tensor x_r, Tensor x_w, Tensor x_k, "
      "Tensor x_v, Tensor x_a, Tensor x_g, Tensor query_start_loc, Tensor "
      "req_id) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor "
      "a0, Tensor a12, Tensor k_a) -> Tensor[]");
  m.def(
      "tmix_kk_a_gate_2d(int B, int T, int C, int H, Tensor k, Tensor k_k, "
      "Tensor a0, Tensor a12, Tensor k_a) -> Tensor[]");
  m.def(
      "tmix_lnx_rkvres_xg(int B, int T, int C, int H, Tensor x, Tensor r, "
      "Tensor k, Tensor v, "
      "Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def(
      "tmix_lnx_rkvres_xg_warp(int B, int T, int C, int H, Tensor x, Tensor "
      "r, Tensor k, Tensor v, Tensor r_k, Tensor weight, Tensor bias, Tensor "
      "g) -> Tensor");
  m.def(
      "tmix_vres_gate(int B, int T, int C, Tensor v, Tensor v_first, Tensor "
      "v0, Tensor v12) -> Tensor");
  m.def(
      "cmix_sparse_down_relu_one(int C, int F, Tensor preact, Tensor value_fc) "
      "-> Tensor");
  m.def(
      "cmix_sparse_down_relu_one_out(int C, int F, Tensor preact, Tensor "
      "value_fc, Tensor(a!) out) -> ()");
  m.def(
      "cmix_sparse_down_relu_rows(int B, int T, int C, int F, Tensor preact, "
      "Tensor value_fc) -> Tensor");
  m.def(
      "cmix_sparse_down_relu_rows_t512(int B, int T, int C, int F, Tensor "
      "preact, Tensor value_fc) -> Tensor");
  m.def(
      "cmix_mix(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor "
      "x_k) -> Tensor");
  m.def(
      "cmix_mix_3d(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor x_k) -> Tensor");
  m.def(
      "cmix_mix_slot(int B, int T, int C, Tensor x, Tensor(a!) shift_state, "
      "Tensor slot_indices, Tensor x_k) -> Tensor");
  m.def(
      "cmix_mix_varlen(int B, int total_tokens, int C, Tensor x, Tensor(a!) "
      "shift_state, Tensor slot_indices, Tensor x_k, Tensor query_start_loc, "
      "Tensor req_id) -> Tensor");
  m.def("relu_square(Tensor x) -> Tensor");
  m.def("act_tanh(Tensor x) -> Tensor");
  m.def("act_sigmoid(Tensor x) -> Tensor");
  m.def("add_vec(int C, Tensor x, Tensor vec) -> Tensor");
  m.def("add_vec_2d(int C, Tensor x, Tensor vec) -> Tensor");
}

TORCH_LIBRARY_IMPL(rwkv7_fast_ops_fp16, CUDA, m) {
  m.impl("tmix_mix6", &tmix_mix6);
  m.impl("tmix_mix6_3d", &tmix_mix6_3d);
  m.impl("tmix_mix6_slot", &tmix_mix6_slot);
  m.impl("tmix_mix6_varlen", &tmix_mix6_varlen);
  m.impl("tmix_kk_a_gate", &tmix_kk_a_gate);
  m.impl("tmix_kk_a_gate_2d", &tmix_kk_a_gate_2d);
  m.impl("tmix_lnx_rkvres_xg", &tmix_lnx_rkvres_xg);
  m.impl("tmix_lnx_rkvres_xg_warp", &tmix_lnx_rkvres_xg_warp);
  m.impl("tmix_vres_gate", &tmix_vres_gate);
  m.impl("cmix_sparse_down_relu_one", &cmix_sparse_down_relu_one);
  m.impl("cmix_sparse_down_relu_one_out", &cmix_sparse_down_relu_one_out);
  m.impl("cmix_sparse_down_relu_rows", &cmix_sparse_down_relu_rows);
  m.impl("cmix_sparse_down_relu_rows_t512", &cmix_sparse_down_relu_rows_t512);
  m.impl("cmix_mix", &cmix_mix);
  m.impl("cmix_mix_3d", &cmix_mix_3d);
  m.impl("cmix_mix_slot", &cmix_mix_slot);
  m.impl("cmix_mix_varlen", &cmix_mix_varlen);
  m.impl("relu_square", &relu_square);
  m.impl("act_tanh", &act_tanh);
  m.impl("act_sigmoid", &act_sigmoid);
  m.impl("add_vec", &add_vec);
  m.impl("add_vec_2d", &add_vec_2d);
}
