cd "$KBENCH_EXPR_PATH/inference/make_datasets/"

python $KBENCH_EXPR_PATH/inference/make_datasets/create_text_dataset.py \
--dataset_name_or_path="$KBENCH_EXPR_PATH/dataset/kernel_bench_data.json" \
--splits="train" \
--output_dir="$KBENCH_EXPR_PATH/dataset_results/parent_commit" \
--retrieval_file="$KBENCH_EXPR_PATH/index_results/parent_commit/kernel_bench_data/file_name_and_contents.retrieval.jsonl" \
--read_linux_from="$KBENCH_EXPR_PATH/index_results/parent_commit/kernel_bench_data/file_name_and_contents_indexes" \
--file_source="bm25" \
--k=3 \
--max_context_len=16000 \
--tokenizer_name="cl100k" \
--problem_statement_max_tokens=10000 \
--commit_type="parent_commit"