from pathlib import Path

from tools.sae_reasoner.visualize import render_feature_report, render_neighbor_report


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
