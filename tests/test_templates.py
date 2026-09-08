from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch

from breeze_infer.templates import (
    _encode_prompt_audio,
    get_template,
    select_template_name,
)


class _FakeAudioTokenizer:
    def __init__(self) -> None:
        self.last_wav: np.ndarray | None = None
        self.last_sr: int | None = None
        self.encode_calls = 0

    def encode(self, wav: np.ndarray, sr: int) -> dict[str, list[np.ndarray]]:
        self.encode_calls += 1
        self.last_wav = wav
        self.last_sr = sr
        return {"audio_codes": [np.zeros((4, 16), dtype=np.int16)]}


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
    ) -> dict[str, list[int] | torch.Tensor]:
        del add_special_tokens
        ids = list(range(2, 2 + len(text)))
        attention_mask = [1] * len(ids)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": attention_mask}

    def decode(self, input_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "x" * len(input_ids)


def test_encode_prompt_audio_reads_with_soundfile_and_downmixes(tmp_path) -> None:
    audio_path = tmp_path / "stereo.wav"
    wav = np.stack(
        [
            np.linspace(-0.5, 0.5, 8, dtype=np.float32),
            np.linspace(0.5, -0.5, 8, dtype=np.float32),
        ],
        axis=1,
    )
    sf.write(audio_path, wav, 24000)
    tokenizer = _FakeAudioTokenizer()

    codes = _encode_prompt_audio(tokenizer, str(audio_path))

    assert isinstance(codes, torch.Tensor)
    assert tuple(codes.shape) == (4, 16)
    assert tokenizer.last_sr == 24000
    assert tokenizer.last_wav is not None
    assert tokenizer.last_wav.shape == (8,)
    np.testing.assert_allclose(tokenizer.last_wav, np.mean(wav, axis=1), atol=1e-4)


@pytest.mark.parametrize(
    ("request_data", "expected"),
    [
        ({"text": "target"}, "tts_plain"),
        ({"text": "target", "instruction": "Speak warmly."}, "tts_instruction"),
        (
            {
                "text": "target",
                "ref_audio_path": "/tmp/ref.wav",
                "ref_text": "reference",
            },
            "ref_clone_tata",
        ),
        (
            {
                "text": "target",
                "instruction": "Speak warmly.",
                "ref_audio_path": "/tmp/ref.wav",
                "ref_text": "reference",
            },
            "ref_edit_tata",
        ),
        (
            {
                "text": "target",
                "instruction": "  ",
                "ref_audio_path": "/tmp/ref.wav",
                "ref_text": "reference",
            },
            "ref_clone_tata",
        ),
        ({"text": "target", "instruction": "  "}, "tts_plain"),
    ],
)
def test_select_template_name_routes_modes(
    request_data: dict[str, str],
    expected: str,
) -> None:
    assert select_template_name(request_data) == expected
    assert get_template(expected).name == expected


@pytest.mark.parametrize(
    "request_data",
    [
        {"text": "target", "ref_audio_path": "/tmp/ref.wav"},
        {"text": "target", "ref_text": "reference"},
    ],
)
def test_select_template_name_rejects_incomplete_reference(
    request_data: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        select_template_name(request_data)


def test_supported_templates_match_pytorch_prompt_layouts() -> None:
    plain_request = {"text": "target", "speaker": "S0"}
    instruction_request = {
        **plain_request,
        "instruction": "Speak warmly.",
    }
    clone_request = {
        **plain_request,
        "ref_audio_path": "/tmp/ref.wav",
        "ref_text": "reference",
    }
    direction_request = {
        **clone_request,
        "instruction": "Speak warmly.",
    }

    assert get_template("tts_plain").build_segments(plain_request) == [
        {"type": "text", "text": "[S0]target"}
    ]
    assert get_template("tts_instruction").build_segments(instruction_request) == [
        {
            "type": "text",
            "text": "[S0]<ins_bos>Speak warmly.<ins_eos>target",
        }
    ]
    assert get_template("ref_clone_tata").build_segments(clone_request) == [
        {"type": "text", "text": "[S0]reference"},
        {
            "type": "audio",
            "append_eos": True,
            "drop_last_frame": False,
            "audio_path": "/tmp/ref.wav",
        },
        {"type": "text", "text": "[S0]target"},
    ]
    assert get_template("ref_edit_tata").build_segments(direction_request) == [
        {"type": "text", "text": "[S0]reference"},
        {
            "type": "audio",
            "append_eos": True,
            "drop_last_frame": False,
            "audio_path": "/tmp/ref.wav",
        },
        {
            "type": "text",
            "text": "[S0]<ins_bos>Speak warmly.<ins_eos>target",
        },
    ]


def test_cfg_branches_match_pytorch_mode_semantics() -> None:
    instruction_request = {
        "text": "target",
        "instruction": "Speak warmly.",
        "speaker": "S0",
    }
    direction_request = {
        **instruction_request,
        "ref_audio_path": "/tmp/ref.wav",
        "ref_text": "reference",
    }

    instruction = get_template("tts_instruction")
    assert instruction.build_negative_segments is not None
    assert instruction.build_negative_segments(instruction_request) == [
        {"type": "text", "text": "[S0]target"}
    ]

    direction = get_template("ref_edit_tata")
    assert direction.build_negative_segments is not None
    assert direction.build_negative_segments(direction_request) == get_template(
        "ref_clone_tata"
    ).build_segments(direction_request)
    assert direction.build_dual_branches is not None
    branches = direction.build_dual_branches(direction_request)
    assert branches["uncond"] == [{"type": "text", "text": "[S0]target"}]
    assert branches["ref"] == direction.build_negative_segments(direction_request)
    assert branches["ins"] == get_template("tts_instruction").build_segments(
        direction_request
    )
