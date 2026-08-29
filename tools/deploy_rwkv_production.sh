#!/usr/bin/env bash
set -euo pipefail

readonly root_dir=/opt/rwkv/vllm
readonly releases_dir="$root_dir/releases"
readonly incoming_dir="$root_dir/incoming"
readonly current_link="$root_dir/current"
readonly previous_link="$root_dir/previous"
readonly models_dir=/srv/rwkv/models
readonly minimum_free_kib=$((50 * 1024 * 1024))

readonly -a services=(
  vllm-rwkv-1_5b.service
  vllm-rwkv-2_9b.service
  vllm-rwkv-7_2b.service
  vllm-rwkv-13_3b.service
)

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

  [[ $(uname -m) == x86_64 ]] || die "production host must be x86_64"
  mapfile -t gpu_names < <(
    nvidia-smi --query-gpu=name --format=csv,noheader
  )
  [[ ${#gpu_names[@]} -eq 4 ]] || die "expected exactly four GPUs"
  for gpu_name in "${gpu_names[@]}"; do
    [[ $gpu_name == "NVIDIA GeForce RTX 4090 D" ]] ||
      die "unexpected GPU: $gpu_name"
  done

  cuda_version=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
  [[ $cuda_version == 13.1 ]] || die "expected CUDA 13.1 driver, got $cuda_version"

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
}

check_models() {
  [[ -f $models_dir/SHA256SUMS ]] || die "$models_dir/SHA256SUMS is missing"
  (
    cd "$models_dir"
    sha256sum --check --strict SHA256SUMS
  )
  check_model rwkv7-g1i-1.5b-20260805-ctx16384 2
  check_model rwkv7-g1i-2.9b-20260805-ctx16384 3
  check_model rwkv7-g1i-7.2b-20260805-ctx16384 6
  check_model rwkv7-g1i-13.3b-20260805-ctx16384 11
}

write_unit() {
  local service=$1
  local description=$2
  local devices=$3
  local model=$4
  local served_name=$5
  local port=$6
  local max_num_seqs=$7
  local memory_utilization=$8
  local parallel_args=${9:-}

  install -m 0644 /dev/stdin "/etc/systemd/system/$service" <<EOF
[Unit]
Description=$description
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rwkv
Group=rwkv
WorkingDirectory=$current_link
Environment=HOME=/home/rwkv
Environment=CUDA_VISIBLE_DEVICES=$devices
Environment=PYTHONUNBUFFERED=1
ExecStart=$current_link/.venv/bin/vllm serve $models_dir/$model \\
  --served-model-name $served_name \\
  --host 127.0.0.1 \\
  --port $port \\
  --dtype float16 \\
  --mamba-ssm-cache-dtype float16 \\
  --max-model-len 16384 \\
  --max-num-seqs $max_num_seqs \\
  --gpu-memory-utilization $memory_utilization \\
  --enable-chunked-prefill \\
  --enable-prefix-caching \\
  --async-scheduling \\
  --enable-auto-tool-choice \\
  --tool-call-parser rwkv$parallel_args
Restart=on-failure
RestartSec=5
TimeoutStopSec=180
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF
}

install_units() {
  write_unit \
    vllm-rwkv-1_5b.service \
    "RWKV7 g1i 1.5B vLLM service" \
    0 rwkv7-g1i-1.5b-20260805-ctx16384 rwkv7-g1i-1.5b \
    18001 1024 0.32
  write_unit \
    vllm-rwkv-2_9b.service \
    "RWKV7 g1i 2.9B vLLM service" \
    0 rwkv7-g1i-2.9b-20260805-ctx16384 rwkv7-g1i-2.9b \
    18002 1024 0.52
  write_unit \
    vllm-rwkv-7_2b.service \
    "RWKV7 g1i 7.2B vLLM service" \
    1 rwkv7-g1i-7.2b-20260805-ctx16384 rwkv7-g1i-7.2b \
    18003 960 0.90
  write_unit \
    vllm-rwkv-13_3b.service \
    "RWKV7 g1i 13.3B vLLM PP2 service" \
    2,3 rwkv7-g1i-13.3b-20260805-ctx16384 rwkv7-g1i-13.3b \
    18004 320 0.90 " --pipeline-parallel-size 2"
  systemctl daemon-reload
  systemctl enable "${services[@]}"
}

stop_services() {
  systemctl stop "${services[@]}" || true
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
  local -a ports=(18001 18002 18003 18004)

  for index in "${!services[@]}"; do
    systemctl start "${services[$index]}"
    if ! probe_service "${ports[$index]}"; then
      systemctl status "${services[$index]}" --no-pager -l || true
      journalctl -u "${services[$index]}" -n 200 --no-pager || true
      return 1
    fi
  done
}

restore_release() {
  local release=$1
  stop_services
  atomic_link "$release" "$current_link"
  start_services || die "failed to restore release $release"
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
  local old_release

  validate_sha "$sha"
  check_host
  check_models
  release=$(stage_release "$sha" "$bundle" "$checksum_file")
  old_release=$(readlink -f "$current_link" 2>/dev/null || true)
  install_units
  stop_services
  atomic_link "$release" "$current_link"

  if ! start_services; then
    if [[ -n $old_release ]]; then
      restore_release "$old_release"
    else
      stop_services
    fi
    die "release $sha failed health checks and was rolled back"
  fi

  if [[ -n $old_release && $old_release != "$release" ]]; then
    atomic_link "$old_release" "$previous_link"
  fi
  prune_releases
  echo "deployed $sha"
}

rollback_release() {
  local old_current
  local target

  require_root
  check_host
  check_models
  old_current=$(readlink -f "$current_link" 2>/dev/null || true)
  target=$(readlink -f "$previous_link" 2>/dev/null || true)
  [[ -n $old_current ]] || die "current release is missing"
  [[ -n $target ]] || die "previous release is missing"
  [[ -x $target/.venv/bin/vllm ]] || die "previous release is invalid"
  restore_release "$target"
  atomic_link "$old_current" "$previous_link"
  echo "rolled back to $(basename "$target")"
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
