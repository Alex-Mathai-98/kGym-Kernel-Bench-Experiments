#!/usr/bin/env python3

"""
Create a dataset for text-to-text training from the raw task instance outputs.
"""

from dotenv import load_dotenv
import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm.auto import tqdm

load_dotenv()
try:
    from create_instance import PROMPT_FUNCTIONS, add_text_inputs
    from tokenize_dataset import TOKENIZER_FUNCS
    from utils import string_to_bool
except:
    from .create_instance import PROMPT_FUNCTIONS, add_text_inputs
    from .tokenize_dataset import TOKENIZER_FUNCS
    from .utils import string_to_bool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl_file(filename):
    if type(filename) == str:
        filename = Path(filename)
    if filename.name.endswith(".jsonl") or filename.name.endswith(".jsonl.all"):
        with open(filename) as f:
            return [json.loads(line) for line in f]
    elif filename.name.endswith(".json"):
        with open(filename) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown file type {filename}")


def instances_generator(files):
    all_data = list()
    for file in tqdm(files, desc="Loading instance files"):
        all_data.extend(load_jsonl_file(file))
    return all_data


def get_training_and_eval_instances(raw_files, test_dataset):
    logger.info("Loading instances")
    raw_instances = list(instances_generator(raw_files))
    final_instances = list(test_dataset["test"])
    eval_repos = {x["repo"] for x in final_instances}
    train_instances = [x for x in raw_instances if x["repo"] not in eval_repos]
    train_instances = list(sorted(train_instances, key=lambda x: x["instance_id"]))
    eval_instances = list(sorted(final_instances, key=lambda x: x["instance_id"]))
    logger.info(f"Found {len(train_instances)} training ids")
    logger.info(f"Found {len(eval_instances)} eval ids")
    return train_instances, eval_instances


def extract_fields(instance):
    instance_id = instance["instance_id"]
    if instance["text_inputs"] is None or instance["patch"] is None:
        print(f"No text for {instance_id}")
        return None
    text_inputs = instance["text_inputs"].strip() + "\n\n"
    if text_inputs is None or instance["patch"] is None:
        print(f"No inputs for {instance_id}")
        return None
    patch = "\n".join([f"<patch>", instance["patch"], "</patch>"])
    return {**instance, "text": text_inputs, "patch": patch}

def pick_K_only(full_data,k) :
    key_list = list(full_data.keys())
    key_list.sort()
    key_list = key_list[:k]
    temp_dict = {}
    for key_ in key_list :
        temp_dict[key_] = full_data[key_]
    full_data = temp_dict
    return full_data

def main(
    dataset_name_or_path,
    splits,
    validation_ratio,
    output_dir,
    retrieval_file,
    read_linux_from,
    prompt_style,
    file_source,
    k,
    max_context_len,
    tokenizer_name,
    problem_statement_max_tokens,
    push_to_hub_user,
    commit_type):
    
    assert(("original_commit" in read_linux_from and \
            "original_commit" in output_dir and \
            "original_commit" in retrieval_file) or 
            ("parent_commit" in read_linux_from and \
            "parent_commit" in output_dir and \
            "parent_commit" in retrieval_file))

    print("Retrieval File : {}".format(retrieval_file))
    if push_to_hub_user is not None:
        hub_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
        assert hub_token is not None, "Must provide HUGGING_FACE_HUB_TOKEN to push to the Hub"
        assert output_dir is None, "Cannot provide output_dir if pushing to the Hub"
    if max_context_len is not None:
        assert tokenizer_name is not None
    if push_to_hub_user is None and not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)
    output_file = f"SWE-bench__{prompt_style}__fs-{file_source}"
    if k is not None:
        assert file_source not in {
            "all",
            "oracle",
        }, "Cannot use max_context_len with oracle or all file sources"
        output_file += f"__k-{k}"
    
    if max_context_len is not None:
        assert file_source not in {
            "all",
            # "oracle",
        }, "Cannot use max_context_len with all file sources"
        # "Cannot use max_context_len with oracle or all file sources"
        assert (
            tokenizer_name is not None
        ), "Must provide tokenizer_name if max_context_len is not None"
        output_file += f"__mcc-{max_context_len}-{tokenizer_name}--{commit_type}"
    if not output_file.endswith(commit_type) :
        output_file += f"--{commit_type}"
    if push_to_hub_user is None:
        output_file = Path(output_dir, output_file)
        if output_file.exists():
            logger.info(f"{output_file.absolute().as_posix()} already exists. Aborting")
            return
    if Path(dataset_name_or_path).exists():
        loaded_data = json.load(open(dataset_name_or_path,"r"))
        if commit_type == "original_commit" :
            print("Assigning 'problem_statement' to 'original_commit_problem_statement'")
            for ele in loaded_data :
                ele["problem_statement"] = ele["original_commit_problem_statement"]
        elif commit_type == "parent_commit" :
            print("Assigning 'problem_statement' to 'parent_commit_problem_statement'")
            for ele in loaded_data :
                ele["problem_statement"] = ele["parent_commit_problem_statement"]
        else :
            raise ValueError("commit_type should be one of original_commit/parent_commit")

        # temporary file where we write the data
        temporary_file_name = dataset_name_or_path.replace(".json","")
        temporary_file_name += "_temp.json"
        json.dump(loaded_data,open(temporary_file_name,"w"),indent=4)
        dataset = load_dataset('json', data_files=temporary_file_name, split=None)
        # remove the temporary file
        os.remove(temporary_file_name)
        # dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)

    split_instances = dict()
    logger.info(f'Found {set(dataset.keys())} splits')
    if set(splits) - set(dataset.keys()) != set():
        raise ValueError(f"Unknown splits {set(splits) - set(dataset.keys())}")
    for split in splits:
        split_instances[split] = {x["instance_id"]: x for x in dataset[split]}
        ##########################################################################
        # split_instances[split] = pick_K_only(split_instances[split],k=50)
        # split_instances[split] = split_instances[split][:10]
        ##########################################################################
        add_text_inputs(
            split_instances[split],
            retrieval_file,
            k,
            prompt_style,
            file_source,
            max_context_len=max_context_len,
            tokenizer_name=tokenizer_name,
            read_linux_from=read_linux_from,
            problem_statement_max_tokens=problem_statement_max_tokens
        )
    columns = [
        "instance_id",
        "text",
        "repo",
        "base_commit",
        "problem_statement",
        "hints_text",
        "created_at",
        "patch",
        "test_patch",
        "version",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "environment_setup_commit",
    ]
    split_data = dict()
    for split in split_instances:
        split_data[split] = {key: list() for key in columns}
        for instance in tqdm(
            split_instances[split].values(), total=len(split_instances[split]), desc=f'Processing {split} instances',
        ):
            datum = extract_fields(instance)
            if datum is None:
                continue
            for key in columns:
                split_data[split][key].append(datum[key] if key in datum else "")
        logger.info(f"Found {len(split_data[split]['instance_id'])} {split} ids")
        split_data[split] = Dataset.from_dict(split_data[split])
    dataset = DatasetDict(split_data)
    if validation_ratio > 0 and "train" in dataset:
        train_val = dataset["train"].train_test_split(
            test_size=validation_ratio,
            seed=42,
        )
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]
    for split in dataset:
        logger.info(f"Found {len(dataset[split])} {split} instances")
    if push_to_hub_user is not None:
        dataset.push_to_hub(f'{push_to_hub_user}/{output_file}', use_auth_token=hub_token)
    else:
        dataset.save_to_disk(output_file)
    logger.info(f"Finsihed saving to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="princeton-nlp/SWE-bench",
        help="Dataset to use for test set from HuggingFace Datasets or path to a save_to_disk directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to use from the dataset.",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.01,
        help="Ratio of the training set to use for validation.",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to the output directory."
    )
    parser.add_argument(
        "--retrieval_file",
        type=str,
        help="Path to the file where the retrieval results are stored.",
    )
    parser.add_argument(
        "--read_linux_from",
        type=str,
        help="Path from where the linux files will be read.",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="style-3",
        choices=PROMPT_FUNCTIONS.keys(),
        help="Prompt style to use. See create_instance.PROMPT_FUNCTIONS for details.",
    )
    parser.add_argument(
        "--file_source",
        type=str,
        default="oracle",
        choices=["oracle", "bm25", "all"],
        help="How to select the files to use in context.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Maximum number of files to use for retrieval.",
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=None,
        help="Maximum number of tokens to use for context.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        choices=TOKENIZER_FUNCS.keys(),
        help="Tokenizer to use for max_context_len. Only needed if max_context_len is specified.",
    )
    parser.add_argument(
        "--problem_statement_max_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to take from the problem statement",
    )
    parser.add_argument(
        "--push_to_hub_user",
        type=str,
        help="Username to use for pushing to the Hub. If not provided, will save to disk.",
    )
    parser.add_argument(
        "--commit_type",
        type=str,
        required=True,
        help="One of parent_commit/original_commit"
    )
    main(**vars(parser.parse_args()))
