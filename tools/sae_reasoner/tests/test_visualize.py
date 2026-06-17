from pathlib import Path

import pytest

from tools.sae_reasoner.visualize import (
    _coerce_background_frames,
    _feature_map_to_thw,
    _select_frame_indices,
    plot_feature_heatmap,
    render_feature_report,
    render_neighbor_report,
)


def test_feature_map_to_thw_promotes_2d_to_single_frame():
    import numpy as np

    arr = _feature_map_to_thw(np.zeros((2, 3)))
    assert arr.shape == (1, 2, 3)  # image map [H,W] -> [1,H,W]
    assert _feature_map_to_thw(np.zeros((4, 2, 3))).shape == (4, 2, 3)
    with pytest.raises(ValueError):
        _feature_map_to_thw(np.zeros((2, 2, 2, 2)))


def test_select_frame_indices_subsamples_evenly():
    assert _select_frame_indices(3, 8) == [0, 1, 2]  # fewer than cap -> all
    picked = _select_frame_indices(100, 5)
    assert picked[0] == 0 and picked[-1] == 99
    assert len(picked) == 5
    assert picked == sorted(picked)


def test_coerce_background_frames_shapes():
    import numpy as np

    assert _coerce_background_frames(None) is None
    single = _coerce_background_frames(np.zeros((8, 8, 3)))
    assert len(single) == 1
    video = _coerce_background_frames(np.zeros((4, 8, 8, 3)))
    assert len(video) == 4


def test_plot_feature_heatmap_smoke():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import numpy as np

    fig = plot_feature_heatmap(np.random.rand(2, 3, 4), title="feat 7")
    # 2 frames -> at least 2 image tiles laid out
    assert len(fig.axes) >= 2
    fig_img = plot_feature_heatmap(np.random.rand(3, 5), frames=np.zeros((20, 20, 3), dtype="uint8"))
    assert len(fig_img.axes) >= 1
    matplotlib.pyplot.close("all")


def test_render_feature_report(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    features.write_text(
        (
            '{"feature_id":3,"activation":2.5,"record_id":"rec","token_index":4,'
            '"prompt":"Describe the scene","media_type":"video","media_path":"s3://bucket/clip.mp4",'
            '"tags":["video","sae_train"],"shard":"000000_rec.pt",'
            '"token_info":{"kind":"video","phase":"prefill","role":"user","token_text":"<|video_pad|>",'
            '"visual_position":{"frame":2,"patch_x":3,"patch_y":4}}}\n'
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.html"

    render_feature_report(features, output, title="Test Features")

    html = output.read_text(encoding="utf-8")
    assert "Test Features" in html
    assert "Feature" in html
    assert "Describe the scene" in html
    assert "patch" in html
    assert "prefill:video" in html
    assert "user" in html
    assert "video" in html


def test_render_neighbor_report(tmp_path: Path):
    neighbors = tmp_path / "neighbors.jsonl"
    neighbors.write_text(
        (
            '{"query":{"record_id":"q","token_index":1,"prompt":"Query prompt",'
            '"media_type":"image","token_info":{"kind":"image","phase":"prefill","role":"user","token_text":"<|image_pad|>",'
            '"visual_position":{"frame":0,"patch_x":2,"patch_y":1}}},'
            '"neighbors":[{"record_id":"n","token_index":2,"prompt":"Neighbor prompt",'
            '"media_type":"image","similarity":0.9,"token_info":{"kind":"text","phase":"decode","role":"assistant","token_text":" robot",'
            '"text_context":"a robot arm"}}]}\n'
        ),
        encoding="utf-8",
    )
    output = tmp_path / "neighbors.html"

    render_neighbor_report(neighbors, output, title="Neighbors")

    html = output.read_text(encoding="utf-8")
    assert "Neighbors" in html
    assert "Query prompt" in html
    assert "Neighbor prompt" in html
    assert "patch" in html
    assert "prefill:image" in html
    assert "decode:text" in html
