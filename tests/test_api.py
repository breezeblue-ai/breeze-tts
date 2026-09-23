from __future__ import annotations

import hashlib
import inspect
import json
import logging
from types import SimpleNamespace

import huggingface_hub
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from breeze_infer import api
from breeze_infer.api import (
    DEFAULT_CFG_SCALE,
    MAX_BATCH_TEXTS,
    _iter_seeded_audio_chunks,
    _pcm16,
    app,
    speech,
)
from breeze_infer.api import (
    MAX_NEW_TOKENS as API_MAX_NEW_TOKENS,
)
from breeze_infer.api import (
    MAX_SEQ_LEN as API_MAX_SEQ_LEN,
)
from infer import MAX_NEW_TOKENS as CLI_MAX_NEW_TOKENS
from infer import MAX_SEQ_LEN as CLI_MAX_SEQ_LEN


def test_api_exposes_only_health_and_streaming_speech() -> None:
    paths = {route.path for route in app.routes if route.path.startswith("/")}

    assert "/health" in paths
    assert "/v1/audio/speech" in paths
    assert "/v1/audio/speech/batch" in paths
    assert "/v1/model" in paths
    assert "/api/ref-audio-codes" not in paths


def test_speech_request_parameters_are_minimal() -> None:
    assert list(inspect.signature(speech).parameters) == [
        "text",
        "instruction",
        "cfg_scale",
        "ref_audio",
        "ref_text",
        "seed",
    ]


def test_api_cfg_defaults_to_one() -> None:
    cfg_parameter = inspect.signature(speech).parameters["cfg_scale"]

    assert DEFAULT_CFG_SCALE == 1.0
    assert cfg_parameter.default.default == 1.0


def test_api_instruction_defaults_to_none() -> None:
    instruction_parameter = inspect.signature(speech).parameters["instruction"]

    assert instruction_parameter.default.default is None


def test_cli_and_api_support_1500_generated_tokens() -> None:
    assert CLI_MAX_NEW_TOKENS == API_MAX_NEW_TOKENS == 1500
    assert CLI_MAX_SEQ_LEN == API_MAX_SEQ_LEN == 2048


def test_pcm16_clips_and_encodes_little_endian() -> None:
    encoded = _pcm16(np.array([-2.0, 0.0, 2.0], dtype=np.float32))

    assert np.frombuffer(encoded, dtype="<i2").tolist() == [-32767, 0, 32767]


def test_streaming_reseeds_immediately_before_model_sampling(monkeypatch) -> None:
    events = []

    class Runtime:
        def iter_audio_chunks(
            self, inputs, *, request_id, seed, token_observer
        ):
            events.append(("sample", inputs, request_id, seed, token_observer))
            yield "chunk"

    monkeypatch.setattr(
        "breeze_infer.api.set_all_seeds",
        lambda seed: events.append(("seed", seed)),
    )

    chunks = list(
        _iter_seeded_audio_chunks(
            Runtime(), {"input_ids": "prepared"}, request_id="request-1", seed=43
        )
    )

    assert chunks == ["chunk"]
    assert events == [
        ("seed", 43),
        ("sample", {"input_ids": "prepared"}, "request-1", 43, None),
    ]


class FakeBatchModel:
    def __init__(self, lengths: list[int] | None = None) -> None:
        self.lengths = lengths
        self.calls = []
        self.head = torch.nn.Linear(2, 2)
        self.weights_during_generate = []
        self.error = None

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.weights_during_generate.append(self.head.weight)
        if self.error is not None:
            raise self.error
        lengths = self.lengths or [2] * kwargs["batch_size"]
        return [torch.full((1, length), 0.5) for length in lengths]


@pytest.fixture
def batch_client(monkeypatch):
    prepared = []

    def fake_prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, **_):
        prepared.append((requests, template))
        return {"batch_size": len(requests)}

    model = FakeBatchModel()
    monkeypatch.setattr(api, "prepare_inputs", fake_prepare_inputs)
    monkeypatch.setattr(api, "set_all_seeds", lambda seed: None)
    monkeypatch.setattr(app.state, "tokenizer", object(), raising=False)
    monkeypatch.setattr(app.state, "audio_tokenizer", object(), raising=False)
    monkeypatch.setattr(app.state, "model", model, raising=False)
    monkeypatch.setattr(
        app.state, "runtime", SimpleNamespace(sample_rate=24000), raising=False
    )
    batch_weight = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    monkeypatch.setattr(
        app.state,
        "batch_head_weights",
        {model.head: batch_weight},
        raising=False,
    )
    client = TestClient(app)
    client.batch_weight = batch_weight
    client.model = model
    client.prepared = prepared
    return client


def _post_batch(client: TestClient, texts, **data):
    if not isinstance(texts, str):
        texts = json.dumps(texts)
    return client.post("/v1/audio/speech/batch", data={"texts": texts, **data})


def test_batch_speech_parameters() -> None:
    assert list(inspect.signature(api.speech_batch).parameters) == [
        "texts",
        "instruction",
        "cfg_scale",
        "ref_audio",
        "ref_text",
        "seed",
        "max_new_tokens",
    ]
    assert MAX_BATCH_TEXTS == 128


@pytest.mark.parametrize(
    "texts",
    [
        "not json",
        {"text": "hello"},
        [],
        ["hello", ""],
        ["hello", "   "],
        ["hello", 3],
        ["hello"] * (MAX_BATCH_TEXTS + 1),
    ],
)
def test_batch_rejects_invalid_texts(batch_client, texts) -> None:
    response = _post_batch(batch_client, texts)

    assert response.status_code == 400
    assert batch_client.model.calls == []
    assert not api._request_lock.locked()


def test_batch_rejects_invalid_cfg_scale(batch_client) -> None:
    response = _post_batch(batch_client, ["hello"], cfg_scale="0")

    assert response.status_code == 400
    assert not api._request_lock.locked()


def test_batch_requires_ref_audio_and_ref_text_together(batch_client) -> None:
    response = _post_batch(batch_client, ["hello"], ref_text="transcript")

    assert response.status_code == 400
    assert not api._request_lock.locked()

    response = batch_client.post(
        "/v1/audio/speech/batch",
        data={"texts": json.dumps(["hello"])},
        files={"ref_audio": ("reference.wav", b"RIFF", "audio/wav")},
    )

    assert response.status_code == 400
    assert not api._request_lock.locked()


def test_batch_returns_409_while_busy(batch_client) -> None:
    assert api._request_lock.acquire(blocking=False)
    try:
        response = _post_batch(batch_client, ["hello"])
    finally:
        api._request_lock.release()

    assert response.status_code == 409
    assert batch_client.model.calls == []


def test_batch_concatenates_segments_with_byte_lengths(batch_client) -> None:
    batch_client.model.lengths = [3, 1]

    response = _post_batch(batch_client, ["one", "two"], seed="7")

    assert response.status_code == 200
    assert response.headers["X-Segment-Bytes"] == "6,2"
    assert response.headers["X-Sample-Rate"] == "24000"
    assert response.headers["X-Sample-Format"] == "s16le"
    assert response.headers["Cache-Control"] == "no-store"
    assert len(response.content) == 8
    assert response.content == _pcm16(np.full(4, 0.5, dtype=np.float32))
    assert not api._request_lock.locked()


def test_batch_generates_eagerly_with_repetition_penalty(batch_client) -> None:
    response = _post_batch(batch_client, ["one", "two"])

    assert response.status_code == 200
    (call,) = batch_client.model.calls
    assert call["output_audio"] is True
    assert call["max_new_tokens"] == api.MAX_NEW_TOKENS
    (processor,) = call["logits_processor"]
    assert isinstance(processor, api.GeneratedTokenRepetitionPenaltyLogitsProcessor)


def test_batch_generates_with_checkpoint_head_weights(batch_client) -> None:
    head = batch_client.model.head
    original = head.weight

    response = _post_batch(batch_client, ["one", "two"])

    assert response.status_code == 200
    assert batch_client.model.weights_during_generate == [batch_client.batch_weight]
    assert head.weight is original


def test_batch_restores_head_weights_when_generate_fails(batch_client) -> None:
    head = batch_client.model.head
    original = head.weight
    batch_client.model.error = RuntimeError("generation failed")

    with pytest.raises(RuntimeError, match="generation failed"):
        _post_batch(batch_client, ["one"])

    assert batch_client.model.weights_during_generate == [batch_client.batch_weight]
    assert head.weight is original
    assert not api._request_lock.locked()


def test_checkpoint_head_weights_survive_in_place_head_casts() -> None:
    model = SimpleNamespace(
        lm_head=torch.nn.Linear(2, 3, bias=False).to(torch.bfloat16),
        depth_decoder=SimpleNamespace(
            codebooks_head=torch.nn.Linear(2, 3, bias=False).to(torch.bfloat16)
        ),
    )
    expected = model.lm_head.weight.detach().clone()

    weights = api._checkpoint_head_weights(model)
    model.lm_head.float()
    model.depth_decoder.codebooks_head.weight.data = (
        model.depth_decoder.codebooks_head.weight.data.float()
    )

    assert set(weights) == {model.lm_head, model.depth_decoder.codebooks_head}
    assert all(weight.dtype == torch.bfloat16 for weight in weights.values())
    assert torch.equal(weights[model.lm_head], expected)
    with api._swapped_weights(weights):
        assert model.lm_head.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.float32


@pytest.mark.parametrize(
    ("requested", "expected"), [("0", 1), ("-5", 1), ("200", 200), ("99999", 1500)]
)
def test_batch_clamps_max_new_tokens(batch_client, requested, expected) -> None:
    response = _post_batch(batch_client, ["hello"], max_new_tokens=requested)

    assert response.status_code == 200
    assert batch_client.model.calls[0]["max_new_tokens"] == expected


def test_batch_selects_template_like_single_request(batch_client) -> None:
    _post_batch(batch_client, ["one", "two"])
    _post_batch(batch_client, ["one"], instruction="  Speak slowly.  ")
    batch_client.post(
        "/v1/audio/speech/batch",
        data={"texts": json.dumps(["one"]), "ref_text": "transcript"},
        files={"ref_audio": ("reference.wav", b"RIFF", "audio/wav")},
    )

    (plain_requests, plain), (instruction_requests, instructed), (ref_requests, ref) = (
        batch_client.prepared
    )
    assert [request["text"] for request in plain_requests] == ["one", "two"]
    assert "instruction" not in plain_requests[0]
    assert plain is api.get_template("tts_plain")
    assert instruction_requests[0]["instruction"] == "Speak slowly."
    assert instructed is api.get_template("tts_instruction")
    assert ref_requests[0]["ref_text"] == "transcript"
    assert ref is api.get_template("ref_clone_tata")


def test_batch_rejects_mismatched_segment_count(batch_client) -> None:
    batch_client.model.lengths = [2]

    with pytest.raises(RuntimeError, match="Batch size mismatch"):
        _post_batch(batch_client, ["one", "two"])
    assert not api._request_lock.locked()


@pytest.fixture
def no_download(monkeypatch):
    def fail(**kwargs):
        raise AssertionError(f"snapshot_download must not be called: {kwargs}")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)


def test_resolve_model_dir_accepts_local_directory(tmp_path, no_download) -> None:
    assert api.resolve_model_dir(str(tmp_path)) == tmp_path


def test_resolve_model_dir_ignores_revision_for_local_directory(
    tmp_path, caplog, no_download
) -> None:
    with caplog.at_level(logging.WARNING, logger=api.__name__):
        assert api.resolve_model_dir(str(tmp_path), "main") == tmp_path

    assert "Ignoring --revision" in caplog.text


def test_resolve_model_dir_downloads_repo_id(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.chdir(tmp_path)

    resolved = api.resolve_model_dir("BreezeBlue/breeze-tts-2", "abc123")

    assert resolved == tmp_path / "snapshot"
    assert calls == [{"repo_id": "BreezeBlue/breeze-tts-2", "revision": "abc123"}]


@pytest.mark.parametrize("model", ["", "/no/such/dir", "not a repo", "a/b/c"])
def test_resolve_model_dir_rejects_invalid_model(
    tmp_path, monkeypatch, model, no_download
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(api.ModelResolutionError):
        api.resolve_model_dir(model)


def test_main_reports_invalid_model_as_usage_error(
    tmp_path, monkeypatch, capsys, no_download
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["breeze_infer.api", "missing-dir"])

    with pytest.raises(SystemExit) as exc_info:
        api.main()

    assert exc_info.value.code == 2
    assert "neither an existing local directory" in capsys.readouterr().err


MODEL_CONFIG = {"codec_config": {"_frame_rate": 12.5}, "model_type": "breeze"}
MODEL_INDEX = {"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}


def _write_checkpoint(model_dir, *, index: bool = True, config=None):
    model_dir.mkdir(parents=True, exist_ok=True)
    config_bytes = json.dumps(config or MODEL_CONFIG).encode()
    index_bytes = json.dumps(MODEL_INDEX).encode()
    (model_dir / "config.json").write_bytes(config_bytes)
    if index:
        (model_dir / "model.safetensors.index.json").write_bytes(index_bytes)
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return config_bytes, index_bytes if index else b""


def test_read_frame_rate(tmp_path) -> None:
    _write_checkpoint(tmp_path)

    assert api.read_frame_rate(tmp_path) == 12.5


def test_read_frame_rate_requires_codec_frame_rate(tmp_path) -> None:
    _write_checkpoint(tmp_path, config={"codec_config": {}})

    with pytest.raises(ValueError, match="_frame_rate"):
        api.read_frame_rate(tmp_path)


@pytest.mark.parametrize("index", [True, False])
def test_model_digest_hashes_config_then_index(tmp_path, index) -> None:
    config_bytes, index_bytes = _write_checkpoint(tmp_path, index=index)

    expected = hashlib.sha256(config_bytes + index_bytes).hexdigest()
    assert api.compute_model_digest(tmp_path) == expected


def test_model_digest_ignores_location_and_weights(tmp_path) -> None:
    _write_checkpoint(tmp_path / "a")
    _write_checkpoint(tmp_path / "b")
    (tmp_path / "b" / "model-00001-of-00001.safetensors").write_bytes(b"other")

    digest = api.compute_model_digest(tmp_path / "a")
    assert digest == api.compute_model_digest(tmp_path / "b")


def test_model_route_reports_checkpoint_facts(tmp_path, monkeypatch) -> None:
    _write_checkpoint(tmp_path)
    settings = api.ApiSettings(
        model=tmp_path,
        fast_all=None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=False,
        fast_depth_decoder=False,
        fast_codec=False,
        model_info=api.build_model_info(tmp_path),
    )
    monkeypatch.setattr(api, "_settings", settings)

    response = TestClient(app).get("/v1/model")

    assert response.status_code == 200
    assert response.json() == {
        "frame_rate": 12.5,
        "model_digest": api.compute_model_digest(tmp_path),
        "max_new_tokens": 1500,
        "max_batch_texts": 128,
    }
