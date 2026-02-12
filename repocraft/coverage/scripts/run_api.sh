#!/bin/bash

# 计算 gpt-4.1 和 gpt-5-mini 的 docs 模式下各 repo 的 coverage

BASE_DIR="/home/v-jianwenluo/temp/repo_encoder_exp"
GT_DIR="/home/v-jianwenluo/temp/CodeAgentTraining/evaluation/coverage/gt_repo_tree"
COVERAGE_SCRIPT="/home/v-jianwenluo/temp/CodeAgentTraining/evaluation/coverage/coverage.py"

# 模型列表
MODELS=("gpt-4.1" "gpt-5-mini")

for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Processing model: $model"
    echo "========================================"

    MODEL_DIR="${BASE_DIR}/${model}/docs"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "Model directory not found: $MODEL_DIR"
        continue
    fi

    # 遍历每个 repo
    for repo_dir in "$MODEL_DIR"/*; do
        if [ ! -d "$repo_dir" ]; then
            continue
        fi

        repo_name=$(basename "$repo_dir")
        input_file="${repo_dir}/checkpoints/new_repo_data.json"
        output_file="${repo_dir}/checkpoints/coverage_result.json"

        # 检查输入文件是否存在
        if [ ! -f "$input_file" ]; then
            echo "[$model/$repo_name] Input file not found: $input_file, skipping..."
            continue
        fi

        # 检查输出文件是否已存在
        if [ -f "$output_file" ]; then
            echo "[$model/$repo_name] Output file already exists: $output_file, skipping..."
            continue
        fi

        echo "[$model/$repo_name] Running coverage evaluation..."
        echo "  Input: $input_file"
        echo "  Output: $output_file"

        python "$COVERAGE_SCRIPT" \
            --input-file "$input_file" \
            --gt-dir "$GT_DIR" \
            --output-file "$output_file"

        if [ $? -eq 0 ]; then
            echo "[$model/$repo_name] Done!"
        else
            echo "[$model/$repo_name] Failed!"
        fi

        echo ""
    done
done

echo "========================================"
echo "All done!"
echo "========================================"
