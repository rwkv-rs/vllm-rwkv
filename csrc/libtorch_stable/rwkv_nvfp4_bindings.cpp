#include <torch/csrc/stable/library.h>

#include "core/registration.h"
#include "libtorch_stable/ops.h"

// The rwkv-nvfp4 profile publishes exactly the Marlin W4A16 and CUTLASS W4A4
// operators consumed by the compressed-tensors NVFP4 linear kernels.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, rwkv_nvfp4_ops) {
  rwkv_nvfp4_ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none, Tensor b_scales, Tensor? a_scales, "
      "Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? g_idx_or_none, "
      "Tensor? perm_or_none, Tensor workspace, int b_type_id, SymInt size_m, "
      "SymInt size_n, SymInt size_k, bool is_k_full, bool use_atomic_add, "
      "bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  rwkv_nvfp4_ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, SymInt size_k, "
      "SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  rwkv_nvfp4_ops.def(
      "cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b, "
      "Tensor block_scale_a, Tensor block_scale_b, Tensor alpha) -> ()");
  rwkv_nvfp4_ops.def(
      "scaled_fp4_quant(Tensor input, Tensor input_scale, bool "
      "is_sf_swizzled_layout) -> (Tensor, Tensor)");
  rwkv_nvfp4_ops.def(
      "scaled_fp4_quant.out(Tensor input, Tensor input_scale, bool "
      "is_sf_swizzled_layout, *, Tensor(a!) output, Tensor(b!) "
      "output_scale) -> ()");
  rwkv_nvfp4_ops.def(
      "cutlass_scaled_mm_supports_fp4(int cuda_device_capability) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, rwkv_nvfp4_ops) {
  rwkv_nvfp4_ops.impl("cutlass_scaled_fp4_mm",
                      TORCH_BOX(&cutlass_scaled_fp4_mm));
  rwkv_nvfp4_ops.impl("scaled_fp4_quant", TORCH_BOX(&scaled_fp4_quant_func));
  rwkv_nvfp4_ops.impl("scaled_fp4_quant.out", TORCH_BOX(&scaled_fp4_quant_out));
}

STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, rwkv_nvfp4_ops) {
  rwkv_nvfp4_ops.impl("cutlass_scaled_mm_supports_fp4",
                      TORCH_BOX(&cutlass_scaled_mm_supports_fp4));
}

REGISTER_EXTENSION(_C_stable_libtorch)
