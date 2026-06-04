import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

from utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_model_and_tokenizer(args: argparse.Namespace):
    pass


def _load_calibration_dataset(args: argparse.Namespace):
    pass


def _build_recipe(args: argparse.Namespace):
    pass


def run(args: argparse.Namespace) -> None:
    logger.info("running template qwen3_1.7b_quant")
