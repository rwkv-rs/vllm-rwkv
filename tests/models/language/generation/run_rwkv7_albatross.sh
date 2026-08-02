#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${repo_root}"

env_file="${RWKV7_ALBATROSS_ENV_FILE:-.env}"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

if [[ "${1:-}" == "--server" ]]; then
  export RWKV7_RUN_SERVER_ALIGNMENT=1
  shift
else
  export RWKV7_RUN_SERVER_ALIGNMENT=0
fi

export ALBATROSS_ROOT="${ALBATROSS_ROOT:-${HOME}/Projects/MachineLearning/albatross}"
export ALBATROSS_IMPL="${ALBATROSS_IMPL:-faster3a_2607}"
export ALBATROSS_REVISION="${ALBATROSS_REVISION:-ee3308f6922e59f2166c7fac3c5a192340a2b48e}"
export VLLM_RWKV7_WKV_MODE="${VLLM_RWKV7_WKV_MODE:-fp32io16}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RAPID_SAMPLER=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export RWKV7_ALBATROSS_MAX_MODEL_LEN="${RWKV7_ALBATROSS_MAX_MODEL_LEN:-1024}"
export RWKV7_ALBATROSS_GPU_MEMORY_UTILIZATION="${RWKV7_ALBATROSS_GPU_MEMORY_UTILIZATION:-0.70}"
export RWKV7_ALBATROSS_ENABLE_FLASHINFER_AUTOTUNE="${RWKV7_ALBATROSS_ENABLE_FLASHINFER_AUTOTUNE:-0}"
export RWKV7_ALBATROSS_EXECUTION_MODES="${RWKV7_ALBATROSS_EXECUTION_MODES:-eager,cudagraph}"
export RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE="${RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE:-1}"
export RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE="${RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE:-1}"
albatross_impl_dir="${ALBATROSS_ROOT}/${ALBATROSS_IMPL}"

missing=()
for name in ALBATROSS_PTH VLLM_RWKV7_MODEL; do
  if [[ -z "${!name:-}" ]]; then
    missing+=("${name}")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf 'Missing required RWKV7 Albatross test environment variables:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf '\nCreate .env or set RWKV7_ALBATROSS_ENV_FILE to a file with:\n' >&2
  printf '  ALBATROSS_PTH=/path/to/rwkv7-g1h-7.2b-20260710-ctx10240.pth\n' >&2
  printf '  VLLM_RWKV7_MODEL=/path/to/matching-rwkv7-hf-artifact\n' >&2
  printf '  # Optional: RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE=2\n' >&2
  printf '  # Optional: RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE=2\n' >&2
  printf '\nUse --server to include the OpenAI server alignment test.\n' >&2
  exit 2
fi

missing_paths=()
for name in ALBATROSS_ROOT ALBATROSS_PTH; do
  if [[ ! -e "${!name}" ]]; then
    missing_paths+=("${name}=${!name}")
  fi
done
if [[ "${VLLM_RWKV7_MODEL}" == *.pth ]]; then
  missing_paths+=("VLLM_RWKV7_MODEL must be a Hugging Face artifact, not .pth")
elif [[ "${VLLM_RWKV7_MODEL}" == /* && ! -d "${VLLM_RWKV7_MODEL}" ]]; then
  missing_paths+=("VLLM_RWKV7_MODEL=${VLLM_RWKV7_MODEL}")
fi
if [[ ! -d "${albatross_impl_dir}" ]]; then
  missing_paths+=("ALBATROSS_ROOT/ALBATROSS_IMPL=${albatross_impl_dir}")
fi

if (( ${#missing_paths[@]} > 0 )); then
  printf 'Missing required RWKV7 Albatross test paths:\n' >&2
  printf '  %s\n' "${missing_paths[@]}" >&2
  printf '\nCheck .env or RWKV7_ALBATROSS_ENV_FILE before running pytest.\n' >&2
  printf 'Use --server to include the OpenAI server alignment test.\n' >&2
  exit 2
fi

python_bin="${PYTHON:-.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  printf 'Python executable not found or not executable: %s\n' "${python_bin}" >&2
  printf 'Set PYTHON=/path/to/python or create the vLLM .venv first.\n' >&2
  exit 2
fi

exec "${python_bin}" -m pytest \
  tests/models/language/generation/test_rwkv7_albatross.py \
  -q -s "$@"
