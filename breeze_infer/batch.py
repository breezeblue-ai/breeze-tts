"""CUDA Graph depth decoding for batched CFG generation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from models.cudagraph.depth_decoder_graph import DepthDecoderGraph


class BatchDepthDecoder:
    """Replace eager per-codebook depth decoding with captured graphs.

    The model's eager CFG depth loop reruns the whole codebook prefix twice per
    codebook. The streaming ``DepthDecoderGraph`` decodes with a static KV cache
    and pads each call to the smallest captured bucket, so one instance serves
    every batch size up to ``max_batch_size`` texts.
    """

    def __init__(self, model: Any, *, max_batch_size: int) -> None:
        bucket_sizes = []
        bucket = 2
        while bucket < 2 * max_batch_size:
            bucket_sizes.append(bucket)
            bucket *= 2
        bucket_sizes.append(bucket)
        depth_decoder = model.depth_decoder
        generation_config = depth_decoder.generation_config
        top_p = generation_config.top_p
        self.graph = DepthDecoderGraph(
            depth_decoder=depth_decoder,
            config=model.config.depth_decoder_config,
            device=str(model.device),
            dtype=next(depth_decoder.model.parameters()).dtype,
            do_sample=bool(generation_config.do_sample),
            top_k=int(generation_config.top_k or 0),
            top_p=1.0 if top_p is None else float(top_p),
            temperature=float(generation_config.temperature),
            num_codebooks=model.config.num_codebooks,
            codec_codebook_size=int(model.config.codec_config.codebook_size),
            batch_size=bucket_sizes[0],
            bucket_sizes=bucket_sizes,
        ).capture()

    def generate_with_cfg(
        self,
        depth_decoder_input_ids: torch.LongTensor,
        cond_backbone_hidden_state: torch.Tensor,
        uncond_backbone_hidden_state: torch.Tensor,
        cfg_scale: float,
    ) -> torch.LongTensor:
        backbone_hidden = torch.cat(
            [cond_backbone_hidden_state, uncond_backbone_hidden_state], dim=0
        )
        first_codebook = depth_decoder_input_ids[:, 1].repeat(2)
        codebooks = self.graph.run(
            backbone_hidden, first_codebook, guidance_scale=cfg_scale
        )
        return torch.cat(
            [depth_decoder_input_ids, codebooks.to(depth_decoder_input_ids.dtype)],
            dim=-1,
        )

    @contextmanager
    def bound(self, model: Any) -> Iterator[None]:
        model._depth_decoder_generate_with_cfg = self.generate_with_cfg
        try:
            yield
        finally:
            del model._depth_decoder_generate_with_cfg
