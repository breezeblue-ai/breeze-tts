from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from models.breeze import BreezeForConditionalGeneration


def get_dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(explicit_device: str | None = None) -> str:
    if explicit_device:
        return explicit_device

    env_device = os.environ.get("BREEZE_DEVICE")
    if env_device:
        return env_device

    _, _, local_rank = get_dist_info()
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    if mps_is_available():
        return "mps"
    return "cpu"


_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def resolve_dtype(device: str, explicit_dtype: str | None = None) -> torch.dtype:
    """Pick the model dtype for a device, honouring an explicit override."""
    name = explicit_dtype or os.environ.get("BREEZE_DTYPE")
    if name:
        try:
            return _DTYPE_BY_NAME[name.strip().lower()]
        except KeyError:
            raise ValueError(
                f"unsupported dtype {name!r}; expected one of "
                f"{sorted(set(_DTYPE_BY_NAME))}"
            ) from None

    if torch.device(device).type == "cpu":
        return torch.float32
    return torch.bfloat16


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if mps_is_available():
        torch.mps.manual_seed(seed)


def update_generation_config_for_breeze(
    model: torch.nn.Module,
    generation_config: dict[str, Any] | None = None,
) -> None:
    generation_config = generation_config or {
        "depth_decoder_do_sample": True,
        "depth_decoder_temperature": 0.9,
        "depth_decoder_top_p": 1.0,
        "depth_decoder_top_k": 50,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "max_new_tokens": 750,
        "temperature": 0.9,
    }

    prefix = "depth_decoder_"
    depth_decoder_attrs = {
        attr[len(prefix) :]: value
        for attr, value in generation_config.items()
        if attr.startswith(prefix)
    }
    vars(model.depth_decoder.generation_config).update(
        {"_from_model_config": False, **depth_decoder_attrs}
    )
    vars(model.generation_config).update(generation_config)


def load_runtime(
    ckpt_dir: Path,
    *,
    device: str,
    attn_implementation: str,
    dtype: torch.dtype | None = None,
) -> tuple[AutoTokenizer, BreezeForConditionalGeneration, Any]:

    if device.startswith("cuda"):
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            rank, world_size, local_rank = get_dist_info()
            raise RuntimeError(
                "Failed to set CUDA device "
                f"device={device} rank={rank} world_size={world_size} local_rank={local_rank} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"device_count={torch.cuda.device_count()}"
            ) from exc
    if dtype is None:
        dtype = resolve_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = BreezeForConditionalGeneration.from_pretrained(
        ckpt_dir,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.to(device).eval()

    from qwen_tts import Qwen3TTSTokenizer

    bundled_audio_tokenizer = ckpt_dir / "audio_tokenizer"
    if not bundled_audio_tokenizer.is_dir():
        raise FileNotFoundError(
            "Bundled audio tokenizer not found at "
            f"{bundled_audio_tokenizer}. The Breeze model package must include "
            "the audio_tokenizer directory."
        )
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(bundled_audio_tokenizer), device_map=device
    )
    return tokenizer, model, audio_tokenizer
