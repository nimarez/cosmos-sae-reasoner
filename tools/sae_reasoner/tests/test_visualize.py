from pathlib import Path

from tools.sae_reasoner.visualize import render_feature_report


def test_render_feature_report(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    features.write_text(
        (
            '{"feature_id":3,"activation":2.5,"record_id":"rec","token_index":4,'
            '"prompt":"Describe the scene","media_type":"video","media_path":"s3://bucket/clip.mp4",'
            '"tags":["video","sae_train"],"shard":"000000_rec.pt",'
            '"token_info":{"kind":"video","token_text":"<|video_pad|>",'
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
    assert "video" in html
