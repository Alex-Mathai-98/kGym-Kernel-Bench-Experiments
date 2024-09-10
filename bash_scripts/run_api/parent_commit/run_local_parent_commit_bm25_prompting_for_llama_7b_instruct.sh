cd "$KBENCH_EXPR_PATH/inference/"
export CUDA_VISIBLE_DEVICES="4,5,6,7"

PATH1="$KBENCH_EXPR_PATH/bash_scripts/run_api/parent_commit"
RETRIEVAL="bm25"
MODEL="CodeLlama-7b-Instruct-hf"

python  $KBENCH_EXPR_PATH/inference/run_llama.py \
--dataset_path="$KBENCH_EXPR_PATH/dataset_results/parent_commit/SWE-bench__style-3__fs-${RETRIEVAL}__k-3__mcc-16000-cl100k--parent_commit" \
--model_name_or_path="meta-llama/$MODEL" \
--output_dir="$KBENCH_EXPR_PATH/prompting_results/parent_commit/$MODEL" \
--split="train" \
--flash_attention="True" > $PATH1/prompting_${RETRIEVAL}_${MODEL}.log 2> $PATH1/prompting_${RETRIEVAL}_${MODEL}.err

python  $KBENCH_EXPR_PATH/inference/run_llama.py \
--dataset_path="$KBENCH_EXPR_PATH/dataset_results/parent_commit/SWE-bench__style-3__fs-${RETRIEVAL}__k-3__mcc-16000-cl100k--parent_commit" \
--model_name_or_path="meta-llama/$MODEL" \
--output_dir="$KBENCH_EXPR_PATH/prompting_results/parent_commit/$MODEL" \
--split="train" \
--temperature=0.5 \
--flash_attention="True" > $PATH1/prompting_${RETRIEVAL}_${MODEL}.log 2> $PATH1/prompting_${RETRIEVAL}_${MODEL}.err


python  $KBENCH_EXPR_PATH/inference/run_llama.py \
--dataset_path="$KBENCH_EXPR_PATH/dataset_results/parent_commit/SWE-bench__style-3__fs-${RETRIEVAL}__k-3__mcc-16000-cl100k--parent_commit" \
--model_name_or_path="meta-llama/$MODEL" \
--output_dir="$KBENCH_EXPR_PATH/prompting_results/parent_commit/$MODEL" \
--split="train" \
--temperature=0.8 \
--flash_attention="True" > $PATH1/prompting_${RETRIEVAL}_${MODEL}.log 2> $PATH1/prompting_${RETRIEVAL}_${MODEL}.err
