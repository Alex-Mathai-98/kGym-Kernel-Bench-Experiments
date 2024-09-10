cd "$KGYM_PATH"

PATH1="$KBENCH_EXPR_PATH/bash_scripts/apply_patches/parent_commit"
RETRIEVAL="bm25"

python $BASE_PATH/run_prompt_predictions.py \
--prediction_file="$KBENCH_EXPR_PATH/prompting_results/parent_commit/gemini-1.5-pro__SWE-bench__style-3__fs-bm25__k-3__mcc-50000-cl100k--parent_commit__train.jsonl" \
--benchmark_dump_path="$KBENCH_PATH" \
--golden_subset="$GOLDEN_SUBSET_PATH" \
--prediction_type="parent_commit" > $PATH1/run_prompts_${RETRIEVAL}_gemini-15-pro.log 2> $PATH1/run_prompts_${RETRIEVAL}_gemini-15-pro.err