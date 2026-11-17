import pytest

from breeze_infer.templates import select_template_name


def test_select_template_name_ignores_whitespace_ref_text():
    # A whitespace-only ref_text is not a usable reference transcript. The
    # instruction check already strips whitespace; ref_text must do the same,
    # otherwise a blank ref_text paired with real audio selects a clone
    # template and downstream prepare_inputs fails on the blank text.
    with pytest.raises(ValueError, match="must be provided together"):
        select_template_name(
            {
                "text": "target",
                "ref_audio_path": "/tmp/ref.wav",
                "ref_text": "   ",
            }
        )
