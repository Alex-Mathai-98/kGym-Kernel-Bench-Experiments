cd "$KBENCH_EXPR_PATH/inference/"
PATH1="$KBENCH_EXPR_PATH/bash_scripts/run_api/parent_commit"
RETRIEVAL="oracle"

python $KBENCH_EXPR_PATH/inference/run_api.py \
--dataset_name_or_path="$KBENCH_EXPR_PATH/dataset_results/parent_commit/SWE-bench__style-3__fs-${RETRIEVAL}__mcc-50000-cl100k--parent_commit" \
--model_name_or_path="gpt-4-turbo-2024-04-09" \
--output_dir="$KBENCH_EXPR_PATH/prompting_results/parent_commit" \
--split="train" \
--commit_type="parent_commit" \
--max_cost=150 > $PATH1/prompting_${RETRIEVAL}_gpt_4.log 2> $PATH1/prompting_${RETRIEVAL}_gpt_4.err
