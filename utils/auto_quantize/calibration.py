"""Text calibration inputs without architecture-specific model imports."""

import json
from pathlib import Path

import torch


def _text(record, tokenizer, text_field=None):
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        raise ValueError("Calibration records must be strings or objects")
    if text_field:
        value = record[text_field]
        if not isinstance(value, str):
            raise ValueError(f"Calibration field {text_field} must contain strings")
        return value
    if isinstance(record.get("text"), str):
        return record["text"]
    if "messages" in record:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False)
        return "\n".join(message["content"] for message in record["messages"])
    if "instruction" in record:
        return "\n".join(record.get(key, "") for key in ("instruction", "input", "output") if record.get(key))
    raise ValueError("No calibration text found; supply text, messages, instruction, or --text-field")


def _records(source, split):
    path = Path(source)
    if path.is_file():
        if path.suffix == ".jsonl":
            with path.open() as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        elif path.suffix == ".json":
            records = json.loads(path.read_text())
            if isinstance(records, dict):
                records = records.get("data")
            if not isinstance(records, list):
                raise ValueError("JSON calibration data must be a list or a {data: [...]} object")
            yield from records
        elif path.suffix == ".txt":
            with path.open() as f:
                paragraph = []
                for line in f:
                    if line.strip():
                        paragraph.append(line.strip())
                    elif paragraph:
                        yield "\n".join(paragraph)
                        paragraph = []
                if paragraph:
                    yield "\n".join(paragraph)
        else:
            raise ValueError("Local calibration files must be .txt, .json or .jsonl")
        return
    from datasets import load_dataset, load_from_disk

    if path.is_dir():
        data = load_from_disk(str(path))
        if hasattr(data, "keys"):
            data = data[split]
    else:
        dataset = "garage-bAInd/Open-Platypus" if source == "open-platypus" else source
        data = load_dataset(dataset, split=split, streaming=True)
    yield from data


def load_calibration(source, tokenizer, *, samples=8, sequence_length=512, split="train", text_field=None):
    if samples < 1 or sequence_length < 2:
        raise ValueError("samples must be positive and sequence_length must be at least 2")
    batches = []
    for record in _records(source, split):
        tokens = tokenizer(_text(record, tokenizer, text_field), return_tensors="pt",
                           add_special_tokens=True)["input_ids"]
        for offset in range(0, tokens.shape[1], sequence_length):
            chunk = tokens[:, offset:offset + sequence_length].cpu().to(torch.long).clone()
            if chunk.shape[1] >= 2:
                batches.append(chunk)
            if len(batches) == samples:
                return batches
    if not batches:
        raise ValueError("Calibration data contains no sequences with at least two tokens")
    return batches
