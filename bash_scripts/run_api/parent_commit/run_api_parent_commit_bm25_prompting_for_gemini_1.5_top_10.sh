cd "$KBENCH_EXPR_PATH/inference/"
PATH1="$KBENCH_EXPR_PATH/bash_scripts/run_api/parent_commit"
MODEL="gemini-1.5-pro"

python $KBENCH_EXPR_PATH/inference/run_api.py \
--dataset_name_or_path="$KBENCH_EXPR_PATH/dataset_results/parent_commit/SWE-bench__style-3__fs-bm25__k-3__mcc-50000-cl100k--parent_commit" \
--model_name_or_path="$MODEL" \
--output_dir="$KBENCH_EXPR_PATH/prompting_results/parent_commit" \
--split="train" \
--commit_type="parent_commit" \
--model_args="top_k=10" \
--max_cost=50 > $PATH1/prompting_BM25_${MODEL}.log 2> $PATH1/prompting_BM25_${MODEL}.err