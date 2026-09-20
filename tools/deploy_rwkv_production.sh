#!/usr/bin/env bash
set -euo pipefail

readonly root_dir=/opt/rwkv/vllm
readonly releases_dir="$root_dir/releases"
readonly incoming_dir="$root_dir/incoming"
readonly current_link="$root_dir/current"
readonly previous_link="$root_dir/previous"
readonly previous_state_dir="$root_dir/previous-state"
readonly models_dir=/srv/rwkv/models
readonly systemd_dir=/etc/systemd/system
readonly router_file=/usr/local/libexec/rwkv_api_router.py
readonly router_service=vllm-rwkv-api-router.service
readonly minimum_free_kib=$((50 * 1024 * 1024))
readonly -a all_services=(
  vllm-rwkv-1_5b.service
  vllm-rwkv-2_9b.service
  vllm-rwkv-7_2b.service
  vllm-rwkv-13_3b.service
)
deployment_role=
listen_host=
declare -a services=()
declare -a ports=()

die() {
  echo "error: $*" >&2
  exit 1
}

require_root() {
  [[ ${EUID} -eq 0 ]] || die "run as root"
}

validate_sha() {
  [[ $1 =~ ^[0-9a-f]{40}$ ]] || die "invalid git SHA: $1"
}

atomic_link() {
  local target=$1
  local link=$2
  local next="${link}.next"
  ln -sfn "$target" "$next"
  mv -Tf "$next" "$link"
}

check_host() {
  local free_kib
  local cuda_version
  local expected_cuda_version
  local expected_gpu_name
  local expected_toolkit_version
  local gpu_name
  local toolkit_version
  local -a gpu_names=()

  [[ $(uname -m) == x86_64 ]] || die "production host must be x86_64"
  mapfile -t gpu_names < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
  )
  [[ ${#gpu_names[@]} -eq 4 ]] || die "expected exactly four GPUs"
  cuda_version=$(
    nvidia-smi |
      sed -En 's/.*CUDA (UMD )?Version: ([0-9.]+).*/\2/p' |
      head -1
  )
  [[ -x /usr/local/cuda/bin/nvcc ]] || die "CUDA toolkit nvcc is missing"
  toolkit_version=$(
    /usr/local/cuda/bin/nvcc --version |
      sed -n 's/.*release \([0-9.]*\),.*/\1/p' |
      tail -1
  )
  case ${gpu_names[0]} in
  "NVIDIA GeForce RTX 4090 D")
    deployment_role=large
    expected_gpu_name="NVIDIA GeForce RTX 4090 D"
    expected_cuda_version=13.1
    expected_toolkit_version=13.1
    listen_host=127.0.0.1
    services=(vllm-rwkv-13_3b.service)
    ports=(18004)
    ;;
  "NVIDIA GeForce RTX 4090")
    deployment_role=small
    expected_gpu_name="NVIDIA GeForce RTX 4090"
    expected_cuda_version=13.2
    expected_toolkit_version=13.0
    listen_host=0.0.0.0
    services=(
      vllm-rwkv-1_5b.service
      vllm-rwkv-2_9b.service
      vllm-rwkv-7_2b.service
    )
    ports=(18001 18002 18003)
    ;;
  *)
    die "unexpected GPU layout: ${gpu_names[*]}"
    ;;
  esac
  for gpu_name in "${gpu_names[@]}"; do
    [[ $gpu_name == "$expected_gpu_name" ]] || die "unexpected GPU: $gpu_name"
  done
  [[ $cuda_version == "$expected_cuda_version" ]] ||
    die "expected CUDA $expected_cuda_version driver, got $cuda_version"
  [[ $toolkit_version == "$expected_toolkit_version" ]] ||
    die "expected CUDA $expected_toolkit_version toolkit, got $toolkit_version"
  command -v c++ >/dev/null || die "C++ compiler is missing"

  mkdir -p "$releases_dir" "$incoming_dir"
  free_kib=$(df -Pk "$root_dir" | awk 'NR == 2 {print $4}')
  [[ $free_kib =~ ^[0-9]+$ ]] || die "cannot determine free disk space"
  ((free_kib >= minimum_free_kib)) || die "less than 50 GiB free under $root_dir"
}

check_model() {
  local model=$1
  local shard_count=$2
  local path="$models_dir/$model"
  local actual_shards

  [[ -f $path/config.json ]] || die "$model is missing config.json"
  [[ -f $path/PROVENANCE.md ]] || die "$model is missing PROVENANCE.md"
  [[ -f $path/model.safetensors.index.json ]] ||
    die "$model is missing model.safetensors.index.json"
  [[ -f $path/tokenizer_config.json ]] || die "$model is missing tokenizer_config.json"
  [[ -f $path/generation_config.json ]] || die "$model is missing generation_config.json"
  [[ -f $path/fake_think_generation_config.json ]] ||
    die "$model is missing fake_think_generation_config.json"
  [[ -f $path/tools_generation_config.json ]] ||
    die "$model is missing tools_generation_config.json"
  actual_shards=$(find "$path" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l)
  [[ $actual_shards -eq $shard_count ]] ||
    die "$model has $actual_shards safetensor shards; expected $shard_count"
  (
    cd "$path"
    awk '
      /^```text$/ { checksums = 1; next }
      /^```$/ && checksums { exit }
      checksums { print }
    ' PROVENANCE.md | sha256sum --check --strict
  )
}

check_models() {
  if [[ $deployment_role == large ]]; then
    check_model rwkv7-g1j-13.3b-20260831-ctx16384 11
  else
    check_model rwkv7-g1j-1.5b-20260831-ctx16384 2
    check_model rwkv7-g1j-2.9b-20260831-ctx16384 3
    check_model rwkv7-g1j-7.2b-20260831-ctx16384 6
  fi
}

write_unit() {
  local release=$1
  local service=$2
  local description=$3
  local devices=$4
  local model=$5
  local served_name=$6
  local port=$7
  local max_num_seqs=$8
  local data_parallel_size=$9

  install -m 0644 /dev/stdin "$systemd_dir/$service" <<EOF
[Unit]
Description=$description
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rwkv
Group=rwkv
WorkingDirectory=$release
Environment=HOME=/home/rwkv
Environment=CUDA_VISIBLE_DEVICES=$devices
Environment=PYTHONUNBUFFERED=1
Environment=XDG_CACHE_HOME=$release/.cache
Environment=PATH=$release/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$release/.venv/bin/vllm serve $models_dir/$model \\
  --served-model-name $served_name \\
  --host $listen_host \\
  --port $port \\
  --dtype float16 \\
  --mamba-ssm-cache-dtype float16 \\
  --max-model-len 16384 \\
  --max-num-seqs $max_num_seqs \\
  --gpu-memory-utilization 0.90 \\
  --data-parallel-size $data_parallel_size \\
  --enable-chunked-prefill \\
  --async-scheduling \\
  --structured-outputs-config.backend xgrammar \\
  --enable-auto-tool-choice \\
  --tool-call-parser rwkv
Restart=on-failure
RestartSec=5
TimeoutStopSec=180
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF
}

install_units() {
  local release=$1

  if [[ $deployment_role == large ]]; then
    write_unit \
      "$release" \
      vllm-rwkv-13_3b.service \
      "RWKV7 g1j 13.3B vLLM DP4 service" \
      0,1,2,3 rwkv7-g1j-13.3b-20260831-ctx16384 rwkv7-g1j-13.3b \
      18004 320 4 || return 1
  else
    write_unit \
      "$release" \
      vllm-rwkv-1_5b.service \
      "RWKV7 g1j 1.5B vLLM service" \
      0 rwkv7-g1j-1.5b-20260831-ctx16384 rwkv7-g1j-1.5b \
      18001 1024 1 || return 1
    write_unit \
      "$release" \
      vllm-rwkv-2_9b.service \
      "RWKV7 g1j 2.9B vLLM service" \
      1 rwkv7-g1j-2.9b-20260831-ctx16384 rwkv7-g1j-2.9b \
      18002 1024 1 || return 1
    write_unit \
      "$release" \
      vllm-rwkv-7_2b.service \
      "RWKV7 g1j 7.2B vLLM DP2 service" \
      2,3 rwkv7-g1j-7.2b-20260831-ctx16384 rwkv7-g1j-7.2b \
      18003 256 2 || return 1
  fi
  systemctl daemon-reload || return 1
  systemctl disable "${all_services[@]}" >/dev/null 2>&1 || true
  systemctl enable "${services[@]}" || return 1
}

stop_services() {
  systemctl stop "${all_services[@]}" || true
}

service_port() {
  case $1 in
  vllm-rwkv-1_5b.service) echo 18001 ;;
  vllm-rwkv-2_9b.service) echo 18002 ;;
  vllm-rwkv-7_2b.service) echo 18003 ;;
  vllm-rwkv-13_3b.service) echo 18004 ;;
  *) return 1 ;;
  esac
}

probe_service() {
  local port=$1
  local deadline=$((SECONDS + 1200))
  local python="$current_link/.venv/bin/python"

  while ((SECONDS < deadline)); do
    if "$python" - "$port" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(
    f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2
) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
    then
      return 0
    fi
    sleep 5
  done
  return 1
}

start_services() {
  local index

  for index in "${!services[@]}"; do
    systemctl start "${services[$index]}"
    if ! probe_service "${ports[$index]}"; then
      systemctl status "${services[$index]}" --no-pager -l || true
      journalctl -u "${services[$index]}" -n 200 --no-pager || true
      return 1
    fi
  done
}

probe_router() {
  probe_service 18000
}

install_release_router() {
  local release=$1

  [[ $deployment_role == large ]] || return 0
  if [[ ! -f $release/temp/rwkv_api_router.py ]]; then
    echo "error: release is missing temp/rwkv_api_router.py" >&2
    return 1
  fi
  install -o root -g root -m 0755 \
    "$release/temp/rwkv_api_router.py" "$router_file" || return 1
  systemctl restart "$router_service" || return 1
  probe_router
}

activate_release() {
  local release=$1

  install_units "$release" || return 1
  stop_services
  atomic_link "$release" "$current_link" || return 1
  start_services || return 1
  install_release_router "$release"
}

snapshot_deployment_state() {
  local state_dir=$1
  local release
  local service

  install -d -m 0700 "$state_dir/units"
  : >"$state_dir/enabled"
  : >"$state_dir/active"
  release=$(readlink -f "$current_link" 2>/dev/null || true)
  printf '%s\n' "$release" >"$state_dir/release"
  for service in "${all_services[@]}"; do
    if [[ -f $systemd_dir/$service ]]; then
      cp -a "$systemd_dir/$service" "$state_dir/units/$service"
    fi
    if systemctl is-enabled --quiet "$service" 2>/dev/null; then
      printf '%s\n' "$service" >>"$state_dir/enabled"
    fi
    if systemctl is-active --quiet "$service"; then
      printf '%s\n' "$service" >>"$state_dir/active"
    fi
  done
  if [[ $deployment_role == large && -f $router_file ]]; then
    cp -a "$router_file" "$state_dir/rwkv_api_router.py"
  fi
  if [[ $deployment_role == large ]] &&
    systemctl is-active --quiet "$router_service"; then
    touch "$state_dir/router-active"
  fi
}

restore_deployment_state() {
  local state_dir=$1
  local port
  local release
  local service

  stop_services
  if [[ $deployment_role == large ]]; then
    systemctl stop "$router_service" || true
  fi
  for service in "${all_services[@]}"; do
    if [[ -f $state_dir/units/$service ]]; then
      install -o root -g root -m 0644 \
        "$state_dir/units/$service" "$systemd_dir/$service" || return 1
    else
      rm -f -- "$systemd_dir/$service"
    fi
  done
  systemctl daemon-reload || return 1
  systemctl disable "${all_services[@]}" >/dev/null 2>&1 || true
  while IFS= read -r service; do
    if [[ -n $service ]]; then
      systemctl enable "$service" || return 1
    fi
  done <"$state_dir/enabled"

  release=$(<"$state_dir/release")
  if [[ -n $release ]]; then
    atomic_link "$release" "$current_link" || return 1
  else
    rm -f -- "$current_link"
  fi
  if [[ $deployment_role == large && -f $state_dir/rwkv_api_router.py ]]; then
    install -o root -g root -m 0755 \
      "$state_dir/rwkv_api_router.py" "$router_file" || return 1
  fi

  while IFS= read -r service; do
    [[ -n $service ]] || continue
    systemctl start "$service" || return 1
    port=$(service_port "$service")
    probe_service "$port" || return 1
  done <"$state_dir/active"
  if [[ -f $state_dir/router-active ]]; then
    systemctl start "$router_service" || return 1
    probe_router || return 1
  fi
}

save_previous_state() {
  local state_dir=$1
  local next_state_dir="$previous_state_dir.next"

  rm -rf -- "$next_state_dir"
  mv "$state_dir" "$next_state_dir"
  rm -rf -- "$previous_state_dir"
  mv "$next_state_dir" "$previous_state_dir"
}

prepare_flashrwkv2() {
  local release=$1
  local cache_root="$release/.cache"
  local cache_dir="$cache_root/torch_extensions"
  local result="$release/flashrwkv2-sm89.json"
  local temporary_result

  install -d -o rwkv -g rwkv -m 0755 "$cache_root"
  temporary_result=$(mktemp "$incoming_dir/flashrwkv2-sm89.XXXXXX")
  if ! runuser -u rwkv -- env \
    HOME=/home/rwkv \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    CUDA_VISIBLE_DEVICES=0 \
    XDG_CACHE_HOME="$cache_root" \
    PATH="$release/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "$release/.venv/bin/python" -m flashrwkv2.compile >"$temporary_result"; then
    rm -f "$temporary_result"
    return 1
  fi
  "$release/.venv/bin/python" - "$temporary_result" "$cache_dir" <<'PY'
import json
import sys
from pathlib import Path

import flashrwkv2
import torch

result = json.loads(Path(sys.argv[1]).read_text())
cache = Path(sys.argv[2]).resolve()
library = Path(result["library"]).resolve()
assert flashrwkv2.__version__ == "0.1.0a13"
assert torch.cuda.get_device_capability() == (8, 9)
assert result["status"] in {"compiled", "cached"}, result
assert result["target"] == "sm89", result
assert library.is_relative_to(cache), (library, cache)
assert library.is_file(), library
PY
  install -o root -g root -m 0644 "$temporary_result" "$result"
  rm -f "$temporary_result"
}

stage_release() {
  local sha=$1
  local bundle=$2
  local checksum_file=$3
  local release="$releases_dir/$sha"
  local staging="$releases_dir/.${sha}.staging"

  [[ -f $bundle ]] || die "bundle does not exist: $bundle"
  [[ -f $checksum_file ]] || die "checksum does not exist: $checksum_file"
  (
    cd "$(dirname "$bundle")"
    sha256sum --check --strict "$(basename "$checksum_file")"
  ) >&2

  if [[ -d $release ]]; then
    [[ $(<"$release/GIT_SHA") == "$sha" ]] ||
      die "existing release has the wrong GIT_SHA"
    prepare_flashrwkv2 "$release" ||
      die "FlashRWKV2 SM89 compilation failed for existing release $sha"
    echo "$release"
    return
  fi

  [[ ! -e $staging ]] || die "staging path already exists: $staging"
  mkdir "$staging"
  tar -xzf "$bundle" -C "$staging"
  [[ -x $staging/.venv/bin/vllm ]] || die "bundle is missing .venv/bin/vllm"
  [[ -x $staging/.venv/bin/python ]] || die "bundle is missing .venv/bin/python"
  [[ -f $staging/GIT_SHA ]] || die "bundle is missing GIT_SHA"
  [[ $(<"$staging/GIT_SHA") == "$sha" ]] || die "bundle SHA does not match $sha"
  chown -R root:root "$staging"
  chmod -R a+rX "$staging"
  mv "$staging" "$release"
  if ! prepare_flashrwkv2 "$release"; then
    rm -rf -- "$release"
    die "FlashRWKV2 SM89 compilation failed for release $sha"
  fi
  echo "$release"
}

prune_releases() {
  local current
  local previous
  local release

  current=$(readlink -f "$current_link")
  previous=$(readlink -f "$previous_link" 2>/dev/null || true)
  while IFS= read -r release; do
    [[ $release == "$current" || $release == "$previous" ]] && continue
    [[ $(basename "$release") =~ ^[0-9a-f]{40}$ ]] ||
      die "refusing to remove unexpected release path: $release"
    rm -rf -- "$release"
  done < <(find "$releases_dir" -mindepth 1 -maxdepth 1 -type d -not -name '.*' | sort)
}

deploy_release() {
  local sha=$1
  local bundle=$2
  local checksum_file=$3
  local release
  local state_dir

  validate_sha "$sha"
  check_host
  check_models
  release=$(stage_release "$sha" "$bundle" "$checksum_file")
  state_dir=$(mktemp -d "$incoming_dir/deployment-state.XXXXXX")
  snapshot_deployment_state "$state_dir"
  if ! activate_release "$release"; then
    if ! restore_deployment_state "$state_dir"; then
      die "release $sha failed and the previous deployment could not be restored"
    fi
    rm -rf -- "$state_dir"
    die "release $sha failed health checks and the previous deployment was restored"
  fi

  if [[ -n $(<"$state_dir/release") ]]; then
    atomic_link "$(<"$state_dir/release")" "$previous_link"
  else
    rm -f -- "$previous_link"
  fi
  save_previous_state "$state_dir"
  prune_releases
  echo "deployed $sha"
}

rollback_release() {
  local current_state_dir

  require_root
  check_host
  [[ -d $previous_state_dir ]] || die "previous deployment state is missing"
  current_state_dir=$(mktemp -d "$incoming_dir/deployment-state.XXXXXX")
  snapshot_deployment_state "$current_state_dir"
  if ! restore_deployment_state "$previous_state_dir"; then
    restore_deployment_state "$current_state_dir" ||
      die "rollback failed and the current deployment could not be restored"
    rm -rf -- "$current_state_dir"
    die "rollback failed; the current deployment was restored"
  fi
  if [[ -n $(<"$current_state_dir/release") ]]; then
    atomic_link "$(<"$current_state_dir/release")" "$previous_link"
  else
    rm -f -- "$previous_link"
  fi
  save_previous_state "$current_state_dir"
  prune_releases
  echo "rolled back deployment"
}

usage() {
  cat >&2 <<EOF
usage:
  $0 deploy <40-char-sha> <bundle.tar.gz> <bundle.tar.gz.sha256>
  $0 rollback
EOF
  exit 2
}

main() {
  require_root
  case ${1:-} in
    deploy)
      [[ $# -eq 4 ]] || usage
      deploy_release "$2" "$3" "$4"
      ;;
    rollback)
      [[ $# -eq 1 ]] || usage
      rollback_release
      ;;
    *) usage ;;
  esac
}

main "$@"
