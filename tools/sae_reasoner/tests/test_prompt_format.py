import pytest
import torch

from tools.sae_reasoner.manifest import ManifestRecord
from tools.sae_reasoner.runtime.cosmos_hf import build_token_map, render_record_prompt, render_text_prompt


class FakeProcessor:
    image_processor = type("FakeImageProcessor", (), {"merge_size": 2})()

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return repr(messages)


class FakeTokenizer:
    all_special_ids = [1, 2]

    def decode(self, ids, skip_special_tokens=False):
        values = {1: "<|im_start|>", 2: "<|im_end|>", 3: " robot", 4: "<|image_pad|>", 5: " pushes"}
        if skip_special_tokens:
            return "".join(values.get(item, str(item)) for item in ids if item not in self.all_special_ids)
        return "".join(values.get(item, str(item)) for item in ids)


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


def test_build_token_map_marks_visual_tokens_and_positions():
    batch = {
        "input_ids": torch.tensor([[1, 3, 4, 4, 4, 4, 5, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }

    token_map, meta = build_token_map(processor=FakeProcessor(), batch=batch, media_type="image")

    assert token_map[0]["kind"] == "special"
    assert token_map[1]["kind"] == "text"
    assert token_map[1]["text_context"] == "<|im_start|> robot<|image_pad|><|image_pad|><|image_pad|><|image_pad|> pushes<|im_end|>"
    assert token_map[2]["kind"] == "image"
    assert token_map[2]["visual_position"] == {
        "frame": 0,
        "patch_y": 0,
        "patch_x": 0,
        "patch_y_range": [0, 2],
        "patch_x_range": [0, 2],
    }
    assert token_map[5]["visual_position"]["patch_x"] == 1
    assert token_map[5]["visual_position"]["patch_y"] == 1
    assert meta["visual_grid"]["merged_grid_thw"] == [1, 2, 2]
