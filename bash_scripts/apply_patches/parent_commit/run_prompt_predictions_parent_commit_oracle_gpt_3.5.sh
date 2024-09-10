cd "$KGYM_PATH"

PATH1="$KBENCH_EXPR_PATH/bash_scripts/apply_patches/parent_commit"
RETRIEVAL="oracle"

python $BASE_PATH/run_prompt_predictions.py \
--prediction_file="$KBENCH_EXPR_PATH/prompting_results/parent_commit/gpt-3.5-turbo-16k-0613__SWE-bench__style-3__fs-${RETRIEVAL}__mcc-16000-cl100k--parent_commit__train.jsonl" \
--benchmark_dump_path="$KBENCH_PATH" \
--golden_subset="$GOLDEN_SUBSET_PATH" \
--prediction_type="parent_commit" > $PATH1/run_prompts_${RETRIEVAL}_gpt_3_5.log 2> $PATH1/run_prompts_${RETRIEVAL}_gpt_3_5.err