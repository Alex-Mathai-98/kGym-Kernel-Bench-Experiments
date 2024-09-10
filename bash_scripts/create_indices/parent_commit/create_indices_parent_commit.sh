cd "$KBENCH_EXPR_PATH/inference/make_datasets/"

# Shard 1 of the dataset
python $KBENCH_EXPR_PATH/inference/make_datasets/bm25_retrieval_modified.py \
--dataset_name_or_path="$KBENCH_EXPR_PATH/dataset/kernel_bench_data.json" \
--output_dir="$KBENCH_EXPR_PATH/index_results/parent_commit" \
--shard_id=0 \
--num_shards=2 \
--splits="train" \
--take_parent_commit="True" \
--linux-path="$LINUX_PATH" > create_parent_indices_shard_0.out 2> create_parent_indices_shard_0.err &

# Shard 2 of the dataset
python $KBENCH_EXPR_PATH/inference/make_datasets/bm25_retrieval_modified.py \
--dataset_name_or_path="$KBENCH_EXPR_PATH/dataset/kernel_bench_data.json" \
--output_dir="$KBENCH_EXPR_PATH/index_results/parent_commit" \
--shard_id=1 \
--num_shards=2 \
--splits="train" \
--take_parent_commit="True" \
--linux-path="$LINUX_PATH_2" > create_parent_indices_shard_1.out 2> create_parent_indices_shard_1.err &
