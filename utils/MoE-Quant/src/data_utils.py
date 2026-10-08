# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import os
import re
from glob import glob
from typing import Optional, List

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


def load_local_dataset(path: str):
    """Load a calibration corpus from a local directory, never touching the Hub.

    Accepts a ``save_to_disk`` directory, the raw layout of a Hub dataset repo
    (``data/*.parquet``; sibling directories such as ``metadata/`` are ignored),
    or any directory of parquet / json / jsonl shards.
    """
    if os.path.exists(os.path.join(path, "dataset_info.json")):
        return load_from_disk(path)
    data_dir = os.path.join(path, "data")
    root = data_dir if os.path.isdir(data_dir) else path
    for pattern, fmt in (("*.parquet", "parquet"), ("*.jsonl", "json"), ("*.json", "json")):
        files = sorted(glob(os.path.join(root, "**", pattern), recursive=True))
        if files:
            return load_dataset(fmt, data_files=files, split="train")
    raise ValueError(f"{path} holds no parquet/json/jsonl shard (looked under {root})")


def split_thought_solution(text: str):
    thought_re = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
    solution_re = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)

    thought = thought_re.search(text).group(1).strip()
    solution = solution_re.search(text).group(1).strip()
    return thought, solution

def prepare_open_thoughts(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42,
    raw_dataset=None
) -> List[torch.Tensor]:
    train_dataset_raw = (
        raw_dataset if raw_dataset is not None
        else load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    )
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Update chat template
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "<think></think>{{render_content(message)}}",
        "{%- set rc = message.get('reasoning_content', '') -%}"
        "<think>{{rc}}</think>{{render_content(message)}}"
    )
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        messages = []
        # add system prompt
        messages.append({"role": "system", "content": example['system']})
        # add dialogue
        for message in example['conversations']:
            role = message["from"]
            if role == "user":
                messages.append({"role": "user", "content": message["value"]})
            else:
                thought, solution = split_thought_solution(message["value"])
                messages.append({"role": "assistant", "content": solution, "reasoning_content": thought})
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset


def prepare_open_platypus(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42,
    raw_dataset=None
) -> List[torch.Tensor]:
    train_dataset_raw = (
        raw_dataset if raw_dataset is not None
        else load_dataset("garage-bAInd/Open-Platypus", split="train")
    )
    if num_calibration_samples:
        train_dataset_raw = train_dataset_raw.shuffle(seed=seed).select(range(num_calibration_samples))
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        messages = [
            {"role": "user", "content": example["instruction"]}, 
            {"role": "assistant", "content":  example["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    train_dataset_raw = train_dataset_raw.map(preprocess)
    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"], 
            padding=False, 
            max_length=max_sequence_length, 
            truncation=True, 
            add_special_tokens=False,
        )
    train_dataset = train_dataset_raw.map(tokenize, remove_columns=train_dataset_raw.column_names)
    train_dataset = [torch.tensor(sample['input_ids']).unsqueeze(0) for sample in train_dataset]
    return train_dataset


def prepare_fineweb_edu(
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    train_dataset_raw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    train_dataset_raw = train_dataset_raw.shuffle(seed=seed, buffer_size=1_000)
    train_dataset = []
    for i, sample in enumerate(train_dataset_raw):
        if i == num_calibration_samples:
            break
        tokenized_sample = tokenizer(
            sample["text"], 
            max_length=max_sequence_length, 
            truncation=True, 
            return_tensors="pt"
        )
        train_dataset.append(tokenized_sample['input_ids'])
    return train_dataset


def prepare_calibration_dataset(
    dataset_name: str, 
    tokenizer: AutoTokenizer, 
    max_sequence_length: int,
    num_calibration_samples: Optional[int] = None,
    seed: int = 42
) -> List[torch.Tensor]:
    # A directory is a locally stored corpus: route on its schema so the local
    # copy goes through the same preprocessing as the matching Hub dataset.
    if os.path.isdir(dataset_name):
        raw_dataset = load_local_dataset(dataset_name)
        columns = set(raw_dataset.column_names)
        if {"system", "conversations"} <= columns:
            return prepare_open_thoughts(
                tokenizer, max_sequence_length, num_calibration_samples, seed, raw_dataset
            )
        if {"instruction", "output"} <= columns:
            return prepare_open_platypus(
                tokenizer, max_sequence_length, num_calibration_samples, seed, raw_dataset
            )
        raise ValueError(
            f"{dataset_name} has columns {sorted(columns)}; expected an OpenThoughts "
            "(system/conversations) or Open-Platypus (instruction/output) corpus"
        )
    if dataset_name == "open-thoughts":
        return prepare_open_thoughts(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "open-platypus":
        return prepare_open_platypus(tokenizer, max_sequence_length, num_calibration_samples, seed)
    if dataset_name == "fineweb-edu":
        return prepare_fineweb_edu(tokenizer, max_sequence_length, num_calibration_samples, seed)
    else:
        raise ValueError("Unknown dataset")
