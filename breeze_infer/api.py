"""Thin streaming API over the PyTorch Breeze inference runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers.generation.logits_process import LogitsProcessorList

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.fast_streaming import (
    FastBreezeStreamingRuntime,
    FastStreamingChunk,
    FastStreamingConfig,
)
from models.logits_process import GeneratedTokenRepetitionPenaltyLogitsProcessor
from models.warmup_profile import load_warmup_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
FAST_CONFIG = REPO_ROOT / "configs" / "fast.json"
DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
OPTIONAL_AUDIO_FILE = File(None)
MAX_BATCH_TEXTS = 128
MODEL_CONFIG_FILE = "config.json"
MODEL_INDEX_FILE = "model.safetensors.index.json"


@dataclass(frozen=True)
class ApiSettings:
    model: Path
    fast_all: bool | None
    fast_text_encoder: bool
    fast_backbone_prefill: bool
    fast_backbone_decode: bool
    fast_depth_decoder: bool
    fast_codec: bool
    model_info: dict[str, float | str | int]


_settings: ApiSettings | None = None
_request_lock = threading.Lock()
logger = logging.getLogger(__name__)


class ModelResolutionError(ValueError):
    """The model argument is neither a local directory nor a Hub repo id."""


def resolve_model_dir(model: str, revision: str | None = None) -> Path:
    """Return a local checkpoint directory for a local path or Hub repo id.

    Repo ids are fetched with ``huggingface_hub.snapshot_download`` into the
    standard Hugging Face cache, so an already downloaded revision is reused.
    """
    if not model.strip():
        raise ModelResolutionError("The model argument is empty.")
    local_dir = Path(model).expanduser()
    if local_dir.is_dir():
        if revision is not None:
            logger.warning(
                "Ignoring --revision %r: %s is a local directory.", revision, local_dir
            )
        return local_dir

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import validate_repo_id

    try:
        if model.count("/") != 1:
            raise HFValidationError("expected '<namespace>/<name>'")
        validate_repo_id(model)
    except HFValidationError as exc:
        raise ModelResolutionError(
            f"{model!r} is neither an existing local directory nor a valid "
            f"Hugging Face repo id such as 'BreezeBlue/breeze-tts-2': {exc}"
        ) from exc
    return Path(snapshot_download(repo_id=model, revision=revision))


def read_frame_rate(model_dir: Path) -> float:
    """Return the codec frame rate from ``codec_config._frame_rate``."""
    config_path = model_dir / MODEL_CONFIG_FILE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        return float(config["codec_config"]["_frame_rate"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{config_path} has no codec_config._frame_rate") from exc


def compute_model_digest(model_dir: Path) -> str:
    """Hash the checkpoint's config and safetensors index, never the weights.

    The digest depends only on file contents, so the same checkpoint yields the
    same value from a local directory or any Hugging Face cache location.
    """
    digest = hashlib.sha256()
    digest.update((model_dir / MODEL_CONFIG_FILE).read_bytes())
    index_path = model_dir / MODEL_INDEX_FILE
    if index_path.is_file():
        digest.update(index_path.read_bytes())
    return digest.hexdigest()


def build_model_info(model_dir: Path) -> dict[str, float | str | int]:
    return {
        "frame_rate": read_frame_rate(model_dir),
        "model_digest": compute_model_digest(model_dir),
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_batch_texts": MAX_BATCH_TEXTS,
    }


def _pcm16(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2", copy=False).tobytes()


def _iter_seeded_audio_chunks(
    runtime: FastBreezeStreamingRuntime,
    inputs: dict[str, object],
    *,
    request_id: str,
    seed: int,
) -> Iterator[FastStreamingChunk]:
    """Start model sampling from the request seed, after all input preparation.

    The response body runs after the endpoint has returned a ``StreamingResponse``.
    Seeding only while preparing the request leaves model sampling vulnerable to
    lazy initialization or unrelated RNG use between preparation and iteration.
    The API is deliberately single-request, so resetting the process generators at
    this boundary gives each request an isolated, reproducible sampling start.
    """
    set_all_seeds(seed)
    token_digest = None
    token_frames = 0
    if os.environ.get("BREEZE_DEBUG_TOKEN_HASH") == "1":
        token_digest = hashlib.sha256()

    def observe_token_frame(frame) -> None:
        nonlocal token_frames
        assert token_digest is not None
        token_digest.update(
            frame.detach().to(device="cpu").contiguous().numpy().tobytes()
        )
        token_frames += 1

    try:
        yield from runtime.iter_audio_chunks(
            inputs,
            request_id=request_id,
            seed=seed,
            token_observer=observe_token_frame if token_digest is not None else None,
        )
    finally:
        if token_digest is not None:
            print(
                "breeze token trace: "
                f"request_id={request_id} seed={seed} frames={token_frames} "
                f"sha256={token_digest.hexdigest()}",
                flush=True,
            )


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "reference.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(
        prefix="breeze_ref_", suffix=suffix, delete=False
    ) as temporary:
        path = Path(temporary.name)
        try:
            payload = await upload.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Reference audio is empty.")
            temporary.write(payload)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return path


def _checkpoint_head_weights(model) -> dict[torch.nn.Module, torch.nn.Parameter]:
    """Snapshot the projection heads before streaming graph setup casts them.

    Streaming graph setup casts ``lm_head`` and the depth decoder's codebook head
    to float32 in place. Eager batch generation runs the heads on bfloat16 hidden
    states, so it swaps these checkpoint-dtype copies in for each call.
    """
    heads = (model.lm_head, model.depth_decoder.codebooks_head)
    return {
        head: torch.nn.Parameter(head.weight.detach().clone(), requires_grad=False)
        for head in heads
    }


@contextmanager
def _swapped_weights(
    weights: dict[torch.nn.Module, torch.nn.Parameter],
) -> Iterator[None]:
    originals = {module: module.weight for module in weights}
    for module, weight in weights.items():
        module.weight = weight
    try:
        yield
    finally:
        for module, weight in originals.items():
            module.weight = weight


def _load_app(app: FastAPI, settings: ApiSettings) -> None:
    tokenizer, model, audio_tokenizer = load_runtime(
        settings.model,
        device=resolve_device(),
        attn_implementation="eager",
    )
    update_generation_config_for_breeze(model)
    batch_head_weights = _checkpoint_head_weights(model)

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=settings.fast_all,
        fast_text_encoder=settings.fast_text_encoder,
        fast_backbone_prefill=settings.fast_backbone_prefill,
        fast_backbone_decode=settings.fast_backbone_decode,
        fast_depth_decoder=settings.fast_depth_decoder,
        fast_codec=settings.fast_codec,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, config, tokenizer=tokenizer
    )
    if runtime.fast_enabled:
        profile = load_warmup_profile(FAST_CONFIG)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms", flush=True)

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.audio_tokenizer = audio_tokenizer
    app.state.runtime = runtime
    app.state.batch_head_weights = batch_head_weights


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    if _settings is None:
        raise RuntimeError("API settings are not initialized")
    _load_app(app, _settings)
    yield


app = FastAPI(title="Breeze TTS API", lifespan=_lifespan)


@app.get("/health")
def health() -> JSONResponse:
    if not hasattr(app.state, "runtime"):
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok", "sample_rate": app.state.runtime.sample_rate})


@app.get("/v1/model")
def model_info() -> JSONResponse:
    if _settings is None:
        raise HTTPException(status_code=503, detail="API settings are not initialized.")
    return JSONResponse(_settings.model_info)


@app.post("/v1/audio/speech")
async def speech(
    text: str = Form(...),
    instruction: str | None = Form(None),
    cfg_scale: float = Form(DEFAULT_CFG_SCALE),
    ref_audio: UploadFile | None = OPTIONAL_AUDIO_FILE,
    ref_text: str = Form(""),
    seed: int = Form(42),
) -> StreamingResponse:
    if not _request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="An inference request is already running."
        )

    reference_path: Path | None = None
    try:
        if not np.isfinite(cfg_scale) or cfg_scale <= 0:
            raise HTTPException(
                status_code=400, detail="cfg_scale must be greater than 0."
            )
        ref_text = ref_text.strip()
        has_reference = ref_audio is not None and bool(ref_audio.filename)
        if has_reference != bool(ref_text):
            raise HTTPException(
                status_code=400,
                detail="ref_audio and ref_text must be provided together or both omitted.",
            )
        if has_reference:
            assert ref_audio is not None
            reference_path = await _save_upload(ref_audio)

        request_id = f"api-{uuid.uuid4().hex}"
        instruction = instruction.strip() if instruction else None
        request = {
            "id": request_id,
            "text": text,
            "speaker": "S0",
        }
        if instruction:
            request["instruction"] = instruction
        if reference_path is not None:
            request["ref_audio_path"] = str(reference_path)
            request["ref_text"] = ref_text
        template_name = select_template_name(request)

        set_all_seeds(seed)
        inputs = prepare_inputs(
            app.state.tokenizer,
            app.state.audio_tokenizer,
            app.state.model,
            [request],
            get_template(template_name),
            guidance_scale=cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
    except Exception:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
        _request_lock.release()
        raise

    def body() -> Iterator[bytes]:
        try:
            for chunk in _iter_seeded_audio_chunks(
                app.state.runtime,
                inputs,
                request_id=request_id,
                seed=seed,
            ):
                pcm = _pcm16(chunk.audio)
                if pcm:
                    yield pcm
        finally:
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)
            _request_lock.release()

    return StreamingResponse(
        body(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(app.state.runtime.sample_rate),
            "X-Sample-Format": "s16le",
            "Cache-Control": "no-store",
        },
    )


def _parse_batch_texts(texts: str) -> list[str]:
    try:
        parsed = json.loads(texts)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"texts must be a JSON array of strings: {exc}"
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise HTTPException(
            status_code=400, detail="texts must be a non-empty JSON array."
        )
    if len(parsed) > MAX_BATCH_TEXTS:
        raise HTTPException(
            status_code=400,
            detail=f"texts has {len(parsed)} entries; the limit is {MAX_BATCH_TEXTS}.",
        )
    if not all(isinstance(item, str) and item.strip() for item in parsed):
        raise HTTPException(
            status_code=400, detail="Every entry in texts must be a non-empty string."
        )
    return parsed


def _generate_batch_pcm(
    texts: list[str],
    *,
    instruction: str | None,
    cfg_scale: float,
    seed: int,
    reference_path: Path | None,
    ref_text: str,
    max_new_tokens: int,
) -> list[bytes]:
    """Synthesize every text in one batched eager ``generate`` call.

    The fast streaming runtime is single-request: its CUDA Graph batch dimension
    is taken by CFG, so it cannot batch texts. Eager generation is slower for a
    single sequence but accepts a real batch, which wins for offline synthesis
    because batch-1 decode re-reads every weight for each frame.
    """
    requests = []
    for index, text in enumerate(texts):
        request = {
            "id": f"batch-{index}",
            "text": text,
            "speaker": "S0",
        }
        if instruction:
            request["instruction"] = instruction
        if reference_path is not None:
            request["ref_audio_path"] = str(reference_path)
            request["ref_text"] = ref_text
        requests.append(request)
    template_name = select_template_name(requests[0])

    set_all_seeds(seed)
    inputs = prepare_inputs(
        app.state.tokenizer,
        app.state.audio_tokenizer,
        app.state.model,
        requests,
        get_template(template_name),
        guidance_scale=cfg_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    logits_processor = LogitsProcessorList(
        [GeneratedTokenRepetitionPenaltyLogitsProcessor(REPETITION_PENALTY)]
    )

    set_all_seeds(seed)
    with torch.inference_mode(), _swapped_weights(app.state.batch_head_weights):
        audio = app.state.model.generate(
            **inputs,
            output_audio=True,
            audio_tokenizer=app.state.audio_tokenizer,
            logits_processor=logits_processor,
            max_new_tokens=max_new_tokens,
        )

    if len(audio) != len(texts):
        raise RuntimeError(
            f"Batch size mismatch: sent {len(texts)} texts, "
            f"got {len(audio)} audio segments."
        )
    return [
        _pcm16(segment.detach().float().cpu().numpy().reshape(-1)) for segment in audio
    ]


@app.post("/v1/audio/speech/batch")
async def speech_batch(
    texts: str = Form(...),
    instruction: str | None = Form(None),
    cfg_scale: float = Form(DEFAULT_CFG_SCALE),
    ref_audio: UploadFile | None = OPTIONAL_AUDIO_FILE,
    ref_text: str = Form(""),
    seed: int = Form(42),
    max_new_tokens: int = Form(MAX_NEW_TOKENS),
) -> Response:
    """Synthesize a JSON array of texts in one batch.

    The body is the concatenated s16le PCM of every segment in request order;
    ``X-Segment-Bytes`` lists each segment's byte length so callers can split it.
    """
    if not _request_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="An inference request is already running."
        )

    reference_path: Path | None = None
    try:
        parsed_texts = _parse_batch_texts(texts)
        if not np.isfinite(cfg_scale) or cfg_scale <= 0:
            raise HTTPException(
                status_code=400, detail="cfg_scale must be greater than 0."
            )
        max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS))
        ref_text = ref_text.strip()
        has_reference = ref_audio is not None and bool(ref_audio.filename)
        if has_reference != bool(ref_text):
            raise HTTPException(
                status_code=400,
                detail="ref_audio and ref_text must be provided together or both omitted.",
            )
        if has_reference:
            assert ref_audio is not None
            reference_path = await _save_upload(ref_audio)

        segments = await run_in_threadpool(
            _generate_batch_pcm,
            parsed_texts,
            instruction=instruction.strip() if instruction else None,
            cfg_scale=cfg_scale,
            seed=seed,
            reference_path=reference_path,
            ref_text=ref_text,
            max_new_tokens=max_new_tokens,
        )
    finally:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
        _request_lock.release()

    return Response(
        content=b"".join(segments),
        media_type="application/octet-stream",
        headers={
            "X-Segment-Bytes": ",".join(str(len(segment)) for segment in segments),
            "X-Sample-Rate": str(app.state.runtime.sample_rate),
            "X-Sample-Format": "s16le",
            "Cache-Control": "no-store",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Breeze TTS 2 streaming inference"
    )
    parser.add_argument(
        "model", help="Local checkpoint directory or Hugging Face repo id"
    )
    parser.add_argument(
        "--revision",
        help="Hugging Face revision for a repo id; ignored for local directories",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-codec", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    try:
        model_dir = resolve_model_dir(args.model, args.revision)
    except ModelResolutionError as exc:
        parser.error(str(exc))

    global _settings
    _settings = ApiSettings(
        model=model_dir,
        fast_all=args.fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
        model_info=build_model_info(model_dir),
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
