"""Text calibration inputs without architecture-specific model imports."""

import json
from pathlib import Path

import torch


def batch_calibration(batches, batch_size=1):
    """Merge independent sequences along batch dimension, never concatenate tokens.

    Right padding gets both an attention mask and ignored labels. All ranks
    retain ceil(sequence_count / batch_size) steps, including a final short batch.
    """
    if batch_size < 1:
        raise ValueError("calibration batch size must be positive")
    batches = list(batches)
    if batch_size == 1:
        return batches
    sequences = []
    for batch in batches:
        data = {"input_ids": batch} if isinstance(batch, torch.Tensor) else batch
        if set(data) - {"input_ids", "attention_mask", "labels"}:
            raise ValueError("Batch merging supports input_ids, attention_mask and labels only")
        ids = data["input_ids"]
        if ids.ndim != 2:
            raise ValueError("Calibration input_ids must have shape [batch, sequence]")
        mask = data.get("attention_mask", torch.ones_like(ids))
        labels = data.get("labels", ids)
        if mask.shape != ids.shape or labels.shape != ids.shape:
            raise ValueError("Calibration masks/labels must match input_ids")
        sequences.extend(zip(ids, mask, labels))
    result = []
    for start in range(0, len(sequences), batch_size):
        group = sequences[start:start + batch_size]
        length = max(ids.numel() for ids, _, _ in group)
        ids = torch.zeros((len(group), length), dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for row, (tokens, valid, target) in enumerate(group):
            size = tokens.numel()
            ids[row, :size] = tokens.cpu()
            mask[row, :size] = valid.cpu()
            labels[row, :size] = target.cpu().masked_fill(~valid.cpu().bool(), -100)
        result.append({"input_ids": ids, "attention_mask": mask, "labels": labels})
    return result


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
