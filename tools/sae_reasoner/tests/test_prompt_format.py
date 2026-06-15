import pytest

from tools.sae_reasoner.manifest import ManifestRecord
from tools.sae_reasoner.runtime.cosmos_hf import render_record_prompt, render_text_prompt


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return repr(messages)


def test_render_record_prompt_rejects_non_chat_format():
    record = ManifestRecord(id="x", media_type="text", prompt="Describe contact.", tags=("text",))

    with pytest.raises(ValueError, match="chat"):
        render_record_prompt(FakeProcessor(), record, prompt_format="raw")


def test_render_chat_prompt_includes_system_prompt():
    rendered = render_text_prompt(
        FakeProcessor(),
        "Describe contact.",
        prompt_format="chat",
        system_prompt="You are terse.",
    )
    assert "system" in rendered
    assert "You are terse." in rendered
    assert "Describe contact." in rendered


def test_render_chat_prompt_preserves_remote_media_uri():
    record = ManifestRecord(
        id="x",
        media_type="image",
        media_path="hf://dataset/org/repo/image.jpg",
        prompt="Describe the image.",
    )

    rendered = render_record_prompt(FakeProcessor(), record, prompt_format="chat")

    assert "hf://dataset/org/repo/image.jpg" in rendered
    assert "hf:/dataset" not in rendered
