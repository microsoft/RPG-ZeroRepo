#!/bin/bash
set -euo pipefail

# ========= Paths =========
BASE_DIR="/home/v-jianwenluo/temp/repo_encoder_exp"
GT_DIR="/home/v-jianwenluo/temp/CodeAgentTraining/evaluation/coverage/gt_repo_tree"
COVERAGE_SCRIPT="/home/v-jianwenluo/temp/CodeAgentTraining/evaluation/coverage/coverage.py"
LOAD_SCRIPT="/home/v-jianwenluo/temp/CodeAgentTraining/evaluation/coverage/load_rpg_to_repo.py"

# ========= 并行度：每次最多同时处理几个 repo =========
PARALLEL=2

# ========= Job matrix =========
# 每个 job: "model|mode|output_filename"
JOBS=(
  "gpt-4.1|ref|coverage_result_3.json"
  "gpt-5-mini|ref|coverage_result_3.json"
  "gpt-5-mini|ablation|coverage_result_3.json"

  "gpt-4.1|ref|coverage_result_3_2.json"
  "gpt-5-mini|ref|coverage_result_3_2.json"
  "gpt-5-mini|ablation|coverage_result_3_2.json"
)

log() { echo -e "$@"; }

# 单个 repo 的执行函数（会被 xargs 并行调用）
run_one_repo() {
  local model="$1"
  local mode="$2"
  local repo_dir="$3"
  local outname="$4"

  local repo_name
  repo_name="$(basename "$repo_dir")"

  local input_file="${repo_dir}/checkpoints/new_repo_data.json"
  local output_file="${repo_dir}/checkpoints/${outname}"

  if [ ! -f "$input_file" ]; then
    echo "[$model/$mode/$repo_name] Input missing: $input_file, skipping..."
    return 0
  fi

  if [ -f "$output_file" ]; then
    echo "[$model/$mode/$repo_name] Output exists: $output_file, skipping..."
    return 0
  fi

  echo "[$model/$mode/$repo_name] Running..."
  echo "  In : $input_file"
  echo "  Out: $output_file"

  python "$COVERAGE_SCRIPT" \
    --input-file "$input_file" \
    --gt-dir "$GT_DIR" \
    --output-file "$output_file"

  echo "[$model/$mode/$repo_name] Done!"
}

export -f run_one_repo
export COVERAGE_SCRIPT GT_DIR

run_coverage() {
  local model="$1"
  local mode="$2"     # ref / ablation
  local outname="$3"

  local model_dir="${BASE_DIR}/${model}/${mode}"
  if [ ! -d "$model_dir" ]; then
    log "[SKIP] Model directory not found: $model_dir"
    return 0
  fi

  log "========================================"
  log "Model: ${model} | Mode: ${mode} | Output: ${outname} | Parallel: ${PARALLEL}"
  log "Dir:  ${model_dir}"
  log "========================================"

  # 找到所有 repo 目录，然后并行跑（最多 PARALLEL 个同时跑）
  # -print0 / -0：防止路径里有空格
  # xargs -n 1 会把目录路径追加到命令末尾，所以要调整参数顺序
  find "$model_dir" -mindepth 1 -maxdepth 1 -type d -print0 \
    | sort -z \
    | xargs -0 -n 1 -P "$PARALLEL" bash -c \
        'run_one_repo "$1" "$2" "$4" "$3"' \
        _ "$model" "$mode" "$outname"
}

# ========= Main =========
python "$LOAD_SCRIPT"

for job in "${JOBS[@]}"; do
  IFS='|' read -r model mode outname <<<"$job"
  run_coverage "$model" "$mode" "$outname"
done

log "========================================"
log "All done!"
log "========================================"