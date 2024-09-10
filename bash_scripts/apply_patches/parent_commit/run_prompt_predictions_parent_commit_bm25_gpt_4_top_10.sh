cd "$KGYM_PATH"

PATH1="$KBENCH_EXPR_PATH/bash_scripts/apply_patches/parent_commit"

python $BASE_PATH/run_prompt_predictions.py \
--prediction_file="$KBENCH_EXPR_PATH/prompting_results/parent_commit/gpt-4-turbo-2024-04-09__SWE-bench__style-3__fs-bm25__k-3__mcc-50000-cl100k--parent_commit__train__top_k=10.jsonl" \
--benchmark_dump_path="$KBENCH_PATH" \
--golden_subset="$GOLDEN_SUBSET_PATH" \
--prediction_type="parent_commit" > $PATH1/run_prompts_gpt_4_top_10.log 2> $PATH1/run_prompts_gpt_4_top_10.err