from tools.sae_reasoner.manifest import ManifestRecord
from tools.sae_reasoner.runtime.cosmos_hf import render_record_prompt, render_text_prompt


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return repr(messages)


def test_render_raw_prompt_without_chat_template():
    record = ManifestRecord(id="x", media_type="text", prompt="Describe contact.", tags=("text",))
    assert render_record_prompt(FakeProcessor(), record, prompt_format="raw") == "Describe contact."


def test_render_raw_prompt_with_system_prompt():
    assert (
        render_text_prompt(
            FakeProcessor(),
            "Describe contact.",
            prompt_format="raw",
            system_prompt="You are terse.",
        )
        == "You are terse.\n\nDescribe contact."
    )


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
