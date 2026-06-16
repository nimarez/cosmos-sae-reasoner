import pytest
import torch
import numpy as np

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


def test_load_video_frames_preserves_metadata(monkeypatch):
    from tools.sae_reasoner.runtime import cosmos_hf

    class FakeCapture:
        def __init__(self, path):
            self.pos = 0
            self.frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(4)]

        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                fake_cv2.CAP_PROP_FRAME_COUNT: 4,
                fake_cv2.CAP_PROP_FPS: 20.0,
                fake_cv2.CAP_PROP_FRAME_WIDTH: 2,
                fake_cv2.CAP_PROP_FRAME_HEIGHT: 2,
            }
            return values.get(prop, 0)

        def set(self, prop, value):
            self.pos = int(value)

        def read(self):
            if self.pos >= len(self.frames):
                return False, None
            frame = self.frames[self.pos]
            self.pos += 1
            return True, frame

        def release(self):
            pass

    class FakeCv2:
        CAP_PROP_FRAME_COUNT = 1
        CAP_PROP_FPS = 2
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_POS_FRAMES = 5
        COLOR_BGR2RGB = 6
        VideoCapture = FakeCapture

        @staticmethod
        def cvtColor(frame, code):
            return frame

    fake_cv2 = FakeCv2()
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)

    frames, metadata = cosmos_hf.load_video_frames_with_metadata("x.mp4", max_frames=2)

    assert frames.shape == (2, 2, 2, 3)
    assert metadata.total_num_frames == 2
    assert metadata.fps == 20.0
    assert metadata.width == 2
    assert metadata.height == 2
    assert metadata.video_backend == "opencv"
    assert metadata.frames_indices == [0, 3]


def test_load_video_frames_falls_back_to_pyav(monkeypatch):
    from tools.sae_reasoner.runtime import cosmos_hf

    def fake_opencv(path, *, frame_limit):
        raise cosmos_hf.RuntimeLoadError("opencv failed")

    def fake_pyav(path, *, frame_limit):
        return np.zeros((1, 2, 2, 3), dtype=np.uint8), type(
            "Meta",
            (),
            {"video_backend": "pyav", "total_num_frames": 1},
        )()

    monkeypatch.setattr(cosmos_hf, "_load_video_frames_with_opencv", fake_opencv)
    monkeypatch.setattr(cosmos_hf, "_load_video_frames_with_pyav", fake_pyav)

    frames, metadata = cosmos_hf.load_video_frames_with_metadata("av1.mp4", max_frames=1)

    assert frames.shape == (1, 2, 2, 3)
    assert metadata.video_backend == "pyav"
