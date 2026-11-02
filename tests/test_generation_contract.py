from types import SimpleNamespace

import pytest
import torch
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.generation.logits_process import (
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)

from models.generation_breeze import BreezeGenerationMixin
from models.logits_process import GeneratedTokenRepetitionPenaltyLogitsProcessor


class Model(BreezeGenerationMixin):
    config = SimpleNamespace(
        num_codebooks=2, codebook_pad_token_id=8,
        codec_config=SimpleNamespace(codebook_size=8),
    )


def processors(penalty=1.0, custom=None, **kwargs):
    config = GenerationConfig(repetition_penalty=penalty, **kwargs)
    config._eos_token_tensor = None
    result = Model()._get_logits_processor(
        config, input_ids_seq_length=2,
        encoder_input_ids=torch.tensor([[262146, 100]]),
        logits_processor=LogitsProcessorList(custom or []), device="cpu",
    )
    assert config.repetition_penalty == penalty
    return result


def test_native_penalty_ignores_prompt_and_only_penalizes_generated_first_codebook():
    configured = processors(2.0)
    scores = torch.tensor([[2., -2., 3.]])
    torch.testing.assert_close(configured(torch.tensor([[262146, 1]]), scores), scores)
    history = torch.tensor([[[1, 2], [0, 2]]])
    torch.testing.assert_close(configured(history, scores), torch.tensor([[1., -4., 3.]]))


@pytest.mark.parametrize("penalty", [1.0, 1.1, 2.0])
def test_native_and_custom_penalty_match(penalty):
    explicit = GeneratedTokenRepetitionPenaltyLogitsProcessor(penalty)
    history = torch.tensor([[[1, 0], [2, 0]]])
    scores = torch.tensor([[2., -2., 3.]])
    torch.testing.assert_close(processors(penalty)(history, scores), explicit(history, scores))


def test_equal_penalties_are_not_applied_twice():
    explicit = GeneratedTokenRepetitionPenaltyLogitsProcessor(2.0)
    configured = processors(2.0, [explicit])
    assert len(configured) == 1
    torch.testing.assert_close(
        configured(torch.tensor([[[1, 0]]]), torch.tensor([[1., 2.]])),
        torch.tensor([[1., 1.]]),
    )


def test_conflicting_custom_penalty_is_rejected():
    with pytest.raises(ValueError, match="Conflicting repetition penalties"):
        processors(1.1, [GeneratedTokenRepetitionPenaltyLogitsProcessor(2.0)])


def test_explicit_hf_penalty_is_adapted():
    configured = processors(custom=[RepetitionPenaltyLogitsProcessor(1.1)])
    assert isinstance(configured[0], GeneratedTokenRepetitionPenaltyLogitsProcessor)
    configured(torch.tensor([[262146]]), torch.ones(1, 8))


@pytest.mark.parametrize("name", ["no_repeat_ngram_size", "encoder_no_repeat_ngram_size"])
def test_ngram_options_are_explicitly_unsupported(name):
    with pytest.raises(ValueError, match=f"{name} is not supported"):
        processors(**{name: 2})


def test_custom_ngram_processor_is_also_rejected():
    with pytest.raises(ValueError, match="N-gram blocking"):
        processors(custom=[NoRepeatNGramLogitsProcessor(2)])


def test_missing_audio_tokenizer_fails_before_parent_generation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("generation must not start without the required audio decoder")
    monkeypatch.setattr(GenerationMixin, "generate", unexpected)
    with pytest.raises(ValueError, match="audio_tokenizer is required"):
        Model().generate(torch.tensor([[262146]]), output_audio=True)


def test_code_only_generation_does_not_require_tokenizer(monkeypatch):
    codes = torch.tensor([[[1, 2], [2, 3]]])
    monkeypatch.setattr(GenerationMixin, "generate", lambda *args, **kwargs: codes)
    assert Model().generate(torch.tensor([[262146]]), output_audio=False) is codes


def test_audio_generation_uses_supplied_decoder(monkeypatch):
    codes = torch.tensor([[[1, 2], [2, 3]]])
    monkeypatch.setattr(GenerationMixin, "generate", lambda *args, **kwargs: codes)
    calls = []
    def decode(payload):
        calls.append(payload["audio_codes"][0])
        return torch.ones(1, 48)
    audio = Model().generate(output_audio=True, audio_tokenizer=SimpleNamespace(decode=decode))
    torch.testing.assert_close(calls[0], codes[0])
    assert audio[0].shape == (48,)
