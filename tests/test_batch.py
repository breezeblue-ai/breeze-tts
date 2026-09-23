from types import SimpleNamespace

import pytest
import torch

from breeze_infer.batch import BatchDepthDecoder


class FakeGraph:
    def __init__(self) -> None:
        self.calls = []

    def run(self, backbone_hidden, first_codebook, *, guidance_scale):
        self.calls.append((backbone_hidden, first_codebook, guidance_scale))
        rows = backbone_hidden.shape[0] // 2
        return torch.arange(rows * 3).view(rows, 3) + 100


def _decoder() -> BatchDepthDecoder:
    decoder = BatchDepthDecoder.__new__(BatchDepthDecoder)
    decoder.graph = FakeGraph()
    return decoder


def test_generate_with_cfg_pairs_cond_and_uncond_rows() -> None:
    decoder = _decoder()
    input_ids = torch.tensor([[0, 5], [0, 7]])
    cond = torch.ones(2, 4)
    uncond = torch.zeros(2, 4)

    sequences = decoder.generate_with_cfg(
        depth_decoder_input_ids=input_ids,
        cond_backbone_hidden_state=cond,
        uncond_backbone_hidden_state=uncond,
        cfg_scale=4.0,
    )

    ((backbone_hidden, first_codebook, guidance_scale),) = decoder.graph.calls
    assert torch.equal(backbone_hidden, torch.cat([cond, uncond]))
    assert first_codebook.tolist() == [5, 7, 5, 7]
    assert guidance_scale == 4.0
    assert sequences.tolist() == [[0, 5, 100, 101, 102], [0, 7, 103, 104, 105]]
    assert sequences.dtype == input_ids.dtype


def test_bound_overrides_cfg_depth_generation_only_inside_context() -> None:
    decoder = _decoder()

    class Model:
        def _depth_decoder_generate_with_cfg(self, *args, **kwargs):
            return "eager"

    model = Model()
    with decoder.bound(model):
        assert model._depth_decoder_generate_with_cfg == decoder.generate_with_cfg
    assert model._depth_decoder_generate_with_cfg() == "eager"

    with pytest.raises(RuntimeError), decoder.bound(model):
        raise RuntimeError("generation failed")
    assert model._depth_decoder_generate_with_cfg() == "eager"


def test_bucket_sizes_cover_paired_rows_for_max_batch(monkeypatch) -> None:
    captured = {}

    class RecordingGraph:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def capture(self):
            return self

    monkeypatch.setattr("breeze_infer.batch.DepthDecoderGraph", RecordingGraph)
    depth_model = torch.nn.Linear(1, 1).to(torch.bfloat16)
    model = SimpleNamespace(
        device=torch.device("cpu"),
        depth_decoder=SimpleNamespace(
            model=depth_model,
            generation_config=SimpleNamespace(
                do_sample=True, top_k=50, top_p=None, temperature=0.9
            ),
        ),
        config=SimpleNamespace(
            depth_decoder_config=object(),
            num_codebooks=16,
            codec_config=SimpleNamespace(codebook_size=2048),
        ),
    )

    BatchDepthDecoder(model, max_batch_size=128)

    assert captured["bucket_sizes"] == [2, 4, 8, 16, 32, 64, 128, 256]
    assert captured["batch_size"] == 2
    assert captured["dtype"] == torch.bfloat16
    assert captured["top_p"] == 1.0
    assert captured["top_k"] == 50
