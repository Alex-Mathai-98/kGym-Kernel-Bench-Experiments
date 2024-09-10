cd "$KGYM_PATH"

PATH1="$KBENCH_EXPR_PATH/bash_scripts/apply_patches/parent_commit"
RETRIEVAL="oracle"

python $BASE_PATH/run_prompt_predictions.py \
--prediction_file="$KBENCH_EXPR_PATH/prompting_results/parent_commit/claude-3-sonnet-20240229__SWE-bench__style-3__fs-oracle__mcc-50000-cl100k--parent_commit__train.jsonl" \
--benchmark_dump_path="$KBENCH_PATH" \
--golden_subset="$GOLDEN_SUBSET_PATH" \
--prediction_type="parent_commit" > $PATH1/run_prompts_${RETRIEVAL}_claude-3-sonnet.log 2> $PATH1/run_prompts_${RETRIEVAL}_claude-3-sonnet.err