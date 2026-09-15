# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys

from utils.logging_config import get_logger

logger = get_logger(__name__)


def run_lm_eval(model_args: str) -> None:
    logger.info("Starting lm_eval with model_args: %s", model_args)

    env = os.environ.copy()
    env["HF_DATASETS_OFFLINE"] = "1"

    default_cache = "./datasets"
    if "HF_DATASETS_CACHE" not in env:
        env["HF_DATASETS_CACHE"] = default_cache

    cache_dir = env["HF_DATASETS_CACHE"] + "/EleutherAI___wikitext_document_level"
    if not os.path.exists(cache_dir):
        logger.error("HF_DATASETS_CACHE directory \"%s\" does not exist. "
                     "Please download 'EleutherAI/wikitext_document_level' to \"%s\", "
                     "or specify via HF_DATASETS_CACHE.",
                     cache_dir, default_cache)
        sys.exit(1)

    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", "wikitext",
        "--batch_size", "1",
    ]

    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        logger.error("lm_eval failed with return code %d", result.returncode)
        sys.exit(result.returncode)

    logger.info("lm_eval completed successfully")
