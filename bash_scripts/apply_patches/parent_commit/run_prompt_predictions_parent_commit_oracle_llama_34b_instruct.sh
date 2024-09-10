cd "$KGYM_PATH"

PATH1="$KBENCH_EXPR_PATH/bash_scripts/apply_patches/parent_commit"
RETRIEVAL="oracle"
MODEL="CodeLlama-34b-Instruct-hf"

python $BASE_PATH/run_prompt_predictions.py \
--prediction_file="$KBENCH_EXPR_PATH/prompting_results/parent_commit/${MODEL}/SWE-bench__style-3__fs-oracle__mcc-16000-cl100k--parent_commit__train__meta-llama__${MODEL}__temp-0.5__top-p-1.0.jsonl" \
--benchmark_dump_path="$KBENCH_PATH" \
--golden_subset="$GOLDEN_SUBSET_PATH" \
--prediction_type="parent_commit" > $PATH1/run_prompts_${RETRIEVAL}_${MODEL}.log 2> $PATH1/run_prompts_${RETRIEVAL}_${MODEL}.err