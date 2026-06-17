from __future__ import annotations

import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .artifacts import ensure_dir, read_jsonl


def _feature_map_to_thw(feature_map: Any):
    """Coerce a feature map to a float numpy array of shape [T, H, W] ([H, W] -> T=1)."""
    import numpy as np

    arr = feature_map.detach().cpu().numpy() if hasattr(feature_map, "detach") else np.asarray(feature_map)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"feature_map must be [T, H, W] or [H, W], got shape {tuple(arr.shape)}")
    return arr


def _select_frame_indices(n_frames: int, max_frames: int | None) -> list[int]:
    """Evenly subsample frame indices down to max_frames (keeps first and last)."""
    if max_frames is None or n_frames <= max_frames:
        return list(range(n_frames))
    import numpy as np

    return sorted({int(round(i)) for i in np.linspace(0, n_frames - 1, max_frames)})


def _coerce_background_frames(frames: Any) -> list[Any] | None:
    """Normalize an optional background (PIL image, [H,W,3], [T,H,W,3], or a list) into a frame list."""
    if frames is None:
        return None
    import numpy as np

    if isinstance(frames, (list, tuple)):
        return [np.asarray(frame) for frame in frames]
    array = np.asarray(frames)
    if array.ndim == 4:
        return [array[i] for i in range(array.shape[0])]
    if array.ndim == 3:
        return [array]
    raise ValueError(f"frames must be a PIL image, [H,W,C], [T,H,W,C], or a list; got shape {tuple(array.shape)}")


def plot_feature_heatmap(
    feature_map: Any,
    *,
    frames: Any = None,
    max_frames: int | None = 8,
    cols: int = 4,
    cmap: str = "inferno",
    alpha: float = 0.5,
    normalize: str | None = "global",
    interpolation: str = "nearest",
    title: str | None = None,
    figsize_per_tile: tuple[float, float] = (3.0, 3.0),
):
    """Render a per-frame heatmap of an SAE feature's spatiotemporal activation map for a notebook.

    feature_map: a [T, H, W] (or [H, W]) array/tensor, e.g. one value of feature_activation_maps().
                 Images are T=1; video is T=num_frames.
    frames:      optional background image(s) to overlay the heatmap on — a PIL image, an [H,W,3]
                 or [T,H,W,3] array, or a list of frames. The low-res grid is upsampled to fit.
    normalize:   "global" (shared color scale + colorbar across frames), "per_frame", or None (raw).

    Returns the matplotlib Figure so the notebook displays it (and you can savefig it).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional notebook dep
        raise RuntimeError("plot_feature_heatmap requires matplotlib; `pip install matplotlib`") from exc
    import numpy as np

    grid = _feature_map_to_thw(feature_map)
    indices = _select_frame_indices(grid.shape[0], max_frames)
    backgrounds = _coerce_background_frames(frames)
    is_video = grid.shape[0] > 1

    global_vmin = global_vmax = None
    if normalize == "global":
        finite = grid[np.isfinite(grid)]
        global_vmin = float(finite.min()) if finite.size else 0.0
        global_vmax = float(finite.max()) if finite.size else 1.0

    count = len(indices)
    ncols = max(1, min(cols, count))
    nrows = math.ceil(count / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_per_tile[0], nrows * figsize_per_tile[1]),
        squeeze=False,
    )

    mappable = None
    for tile, frame_idx in enumerate(indices):
        ax = axes[tile // ncols][tile % ncols]
        heat = grid[frame_idx]
        vmin, vmax = global_vmin, global_vmax
        if normalize == "per_frame":
            finite = heat[np.isfinite(heat)]
            vmin = float(finite.min()) if finite.size else 0.0
            vmax = float(finite.max()) if finite.size else 1.0
        background = backgrounds[frame_idx] if backgrounds is not None and frame_idx < len(backgrounds) else None
        if background is not None:
            ax.imshow(background)
            height, width = int(background.shape[0]), int(background.shape[1])
            mappable = ax.imshow(
                heat, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                extent=[0, width, height, 0], interpolation=interpolation, aspect="auto",
            )
        else:
            mappable = ax.imshow(heat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
        ax.set_title(f"frame {frame_idx}" if is_video else "feature map", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for empty in range(count, nrows * ncols):
        axes[empty // ncols][empty % ncols].axis("off")

    if mappable is not None and normalize == "global":
        fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.8, label="feature activation")
    if title:
        fig.suptitle(title)
    return fig


def render_feature_report(features_path: Path, output: Path, *, title: str = "Cosmos SAE Feature Browser") -> None:
    rows = read_jsonl(features_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("feature_id", "unknown"))].append(row)
    features = []
    for feature_id, examples in sorted(grouped.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        examples.sort(key=lambda row: float(row.get("activation_score", abs(float(row.get("activation", 0.0))))), reverse=True)
        max_activation = max((abs(float(row.get("activation", 0.0))) for row in examples), default=0.0)
        features.append(
            {
                "feature_id": feature_id,
                "max_activation": max_activation,
                "breakdown": feature_breakdown(examples),
                "examples": examples,
            }
        )
    ensure_dir(output.parent)
    output.write_text(render_html(features, title=title), encoding="utf-8")


def render_neighbor_report(neighbors_path: Path, output: Path, *, title: str = "Cosmos Activation Nearest Neighbors") -> None:
    rows = read_jsonl(neighbors_path)
    ensure_dir(output.parent)
    output.write_text(render_neighbor_html(rows, title=title), encoding="utf-8")


def feature_breakdown(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return token_breakdown(examples)


def token_breakdown(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for row in examples:
        token = row.get("token_info") or {}
        phase = token.get("phase") or "unknown"
        kind = token.get("kind") or "unknown"
        counts[f"{phase}:{kind}"] += 1
    total = max(1, len(examples))
    return [
        {"label": label, "count": count, "fraction": count / total}
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def neighbor_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [row.get("query") or {} for row in rows]
    neighbors = [neighbor for row in rows for neighbor in row.get("neighbors", [])]
    return {
        "query_count": len(queries),
        "neighbor_count": len(neighbors),
        "query_breakdown": token_breakdown(queries),
        "neighbor_breakdown": token_breakdown(neighbors),
    }


def render_html(features: list[dict[str, Any]], *, title: str) -> str:
    data = json.dumps(features, ensure_ascii=True)
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #181816;
      --muted: #686862;
      --line: #d8d7cf;
      --accent: #0f6f5f;
      --accent-soft: #dff0eb;
      --warn: #9b5b12;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 247, 244, 0.94);
      backdrop-filter: blur(8px);
      padding: 14px 20px;
      display: flex;
      gap: 16px;
      align-items: center;
    }}
    h1 {{
      font-size: 18px;
      line-height: 1.2;
      margin: 0;
      white-space: nowrap;
    }}
    input {{
      width: min(520px, 100%);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 11px;
      font-size: 14px;
      background: var(--panel);
      color: var(--ink);
    }}
    main {{
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      min-height: calc(100vh - 58px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      padding: 12px;
      overflow: auto;
      max-height: calc(100vh - 58px);
      position: sticky;
      top: 58px;
    }}
    .feature-button {{
      width: 100%;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      color: var(--ink);
      padding: 8px 9px;
      font: inherit;
      text-align: left;
      cursor: pointer;
    }}
    .feature-button:hover, .feature-button.active {{
      border-color: var(--line);
      background: var(--panel);
    }}
    .count {{
      color: var(--muted);
      font-size: 12px;
    }}
    .content {{
      padding: 22px;
      max-width: 1180px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }}
    .metric, .example {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{
      padding: 12px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .metric .value {{
      font-size: 22px;
      font-weight: 650;
    }}
    .examples {{
      display: grid;
      gap: 12px;
    }}
    .example {{
      padding: 14px;
    }}
    .example-head {{
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }}
    .activation {{
      min-width: 170px;
      display: grid;
      gap: 4px;
    }}
    .bar {{
      height: 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      overflow: hidden;
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}
    .prompt {{
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 14px;
    }}
    .meta {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      background: #fbfbf8;
    }}
	    .path {{
      margin-top: 10px;
      color: var(--warn);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
	      overflow-wrap: anywhere;
	    }}
	    .breakdown {{
	      display: grid;
	      gap: 6px;
	      margin-bottom: 18px;
	    }}
	    .breakdown-row {{
	      display: grid;
	      grid-template-columns: 150px minmax(0, 1fr) 64px;
	      gap: 8px;
	      align-items: center;
	      color: var(--muted);
	      font-size: 12px;
	    }}
    @media (max-width: 820px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ position: static; max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }}
      .summary {{ grid-template-columns: 1fr; }}
      header {{ align-items: stretch; flex-direction: column; }}
      h1 {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <input id="search" placeholder="Filter by feature, prompt, tag, or record id">
  </header>
  <main>
    <aside id="feature-list"></aside>
    <section class="content" id="content"></section>
  </main>
  <script>
    const FEATURES = {data};
    const state = {{ selected: FEATURES[0]?.feature_id ?? null, query: "" }};
    const listEl = document.getElementById("feature-list");
    const contentEl = document.getElementById("content");
    const searchEl = document.getElementById("search");

    function textOf(value) {{
      if (value == null) return "";
      if (Array.isArray(value)) return value.join(" ");
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
    }}

    function matches(feature) {{
      const q = state.query.trim().toLowerCase();
      if (!q) return true;
      if (String(feature.feature_id).toLowerCase().includes(q)) return true;
      return feature.examples.some((row) => [
        row.prompt, row.record_id, row.media_type, row.media_path, row.tags,
        row.token_info?.kind, row.token_info?.token_text, row.token_info?.text_context
      ].map(textOf).join(" ").toLowerCase().includes(q));
    }}

    function esc(value) {{
      return textOf(value).replace(/[&<>"']/g, (c) => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
    }}

    function renderList() {{
      const visible = FEATURES.filter(matches);
      if (!visible.find((f) => f.feature_id === state.selected)) {{
        state.selected = visible[0]?.feature_id ?? null;
      }}
      listEl.innerHTML = visible.map((feature) => `
        <button class="feature-button ${{feature.feature_id === state.selected ? "active" : ""}}" data-feature="${{esc(feature.feature_id)}}">
          <span>Feature ${{esc(feature.feature_id)}}</span>
          <span class="count">${{feature.examples.length}}</span>
        </button>
      `).join("");
      for (const button of listEl.querySelectorAll("button")) {{
        button.addEventListener("click", () => {{
          state.selected = button.dataset.feature;
          render();
        }});
      }}
    }}

    function renderContent() {{
      const feature = FEATURES.find((item) => item.feature_id === state.selected);
      if (!feature) {{
        contentEl.innerHTML = "<p>No matching features.</p>";
        return;
      }}
      const max = Math.max(1e-9, feature.max_activation || 0);
      contentEl.innerHTML = `
        <div class="summary">
          <div class="metric"><div class="label">Feature</div><div class="value">${{esc(feature.feature_id)}}</div></div>
	          <div class="metric"><div class="label">Examples</div><div class="value">${{feature.examples.length}}</div></div>
	          <div class="metric"><div class="label">Max |activation|</div><div class="value">${{Number(feature.max_activation || 0).toFixed(3)}}</div></div>
	        </div>
	        <div class="breakdown">
	          ${{(feature.breakdown || []).map((row) => `
	            <div class="breakdown-row">
	              <span>${{esc(row.label)}}</span>
	              <div class="bar"><span style="width:${{Math.max(2, Math.min(100, 100 * Number(row.fraction || 0)))}}%"></span></div>
	              <span>${{esc(row.count)}} ex</span>
	            </div>
	          `).join("")}}
	        </div>
	        <div class="examples">
          ${{feature.examples.map((row) => renderExample(row, max)).join("")}}
        </div>
      `;
    }}

    function renderExample(row, max) {{
      const activation = Number(row.activation || 0);
      const width = Math.max(2, Math.min(100, 100 * Math.abs(activation) / max));
      const tags = Array.isArray(row.tags) ? row.tags : [];
      const mediaPath = row.media_path || row.metadata?.source_uri || "";
      const token = row.token_info || {{}};
      const pos = token.visual_position || {{}};
      const tokenDetail = token.kind === "text"
        ? `${{esc(token.token_text || "")}}${{token.text_context ? `<div class="path">${{esc(token.text_context)}}</div>` : ""}}`
        : token.visual_position
          ? `frame ${{esc(pos.frame)}} - patch (${{esc(pos.patch_x)}}, ${{esc(pos.patch_y)}})`
          : esc(token.token_text || "");
      return `
        <article class="example">
          <div class="example-head">
            <strong>${{esc(row.record_id || "(unknown record)")}}</strong>
            <div class="activation">
              <span>${{activation.toFixed(4)}}</span>
              <div class="bar"><span style="width:${{width}}%"></span></div>
            </div>
          </div>
          <div class="prompt">${{esc(row.prompt || "")}}</div>
          <div class="meta">
            <span class="pill">${{esc(row.media_type || "unknown")}}</span>
	            <span class="pill">token ${{esc(row.token_index)}}</span>
	            ${{token.kind ? `<span class="pill">${{esc(token.kind)}}</span>` : ""}}
	            ${{token.phase ? `<span class="pill">${{esc(token.phase)}}</span>` : ""}}
	            ${{token.role ? `<span class="pill">${{esc(token.role)}}</span>` : ""}}
	            <span class="pill">${{esc(row.shard || "")}}</span>
            ${{tags.map((tag) => `<span class="pill">${{esc(tag)}}</span>`).join("")}}
          </div>
          ${{token.kind ? `<div class="path">${{tokenDetail}}</div>` : ""}}
          ${{mediaPath ? `<div class="path">${{esc(mediaPath)}}</div>` : ""}}
        </article>
      `;
    }}

    function render() {{
      renderList();
      renderContent();
    }}
    searchEl.addEventListener("input", () => {{
      state.query = searchEl.value;
      render();
    }});
    render();
  </script>
</body>
</html>
"""


def render_neighbor_html(rows: list[dict[str, Any]], *, title: str) -> str:
    data = json.dumps(rows, ensure_ascii=True)
    summary_data = json.dumps(neighbor_summary(rows), ensure_ascii=True)
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      --bg: #f6f7f7;
      --panel: #ffffff;
      --ink: #161819;
      --muted: #657074;
      --line: #d7dddf;
      --accent: #15616d;
      --accent-soft: #dceff1;
      --warn: #8c5a14;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 247, 0.94);
      backdrop-filter: blur(8px);
      padding: 14px 20px;
      display: flex;
      gap: 16px;
      align-items: center;
    }}
    h1 {{ font-size: 18px; margin: 0; white-space: nowrap; }}
    input {{
      width: min(560px, 100%);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 11px;
      font-size: 14px;
      background: var(--panel);
      color: var(--ink);
    }}
    main {{ padding: 18px; display: grid; gap: 16px; max-width: 1300px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .metric .label {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .metric .value {{ font-size: 20px; font-weight: 650; }}
    .breakdown {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .group {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .query {{
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      padding: 10px;
      border-radius: 6px;
      margin-bottom: 12px;
    }}
    .neighbors {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; }}
    .neighbor {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; }}
    .meta {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; color: var(--muted); font-size: 12px; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 3px 7px; background: #fbfbf8; }}
    .prompt {{ margin-top: 8px; white-space: pre-wrap; line-height: 1.35; font-size: 13px; }}
    .path {{
      margin-top: 8px;
      color: var(--warn);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .score {{ color: var(--accent); font-weight: 650; }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <input id="search" placeholder="Filter by record, prompt, token kind, tag, or path">
  </header>
  <main id="content"></main>
  <script>
    const ROWS = {data};
    const SUMMARY = {summary_data};
    const contentEl = document.getElementById("content");
    const searchEl = document.getElementById("search");

    function textOf(value) {{
      if (value == null) return "";
      if (Array.isArray(value)) return value.join(" ");
      if (typeof value === "object") return JSON.stringify(value);
      return String(value);
    }}

    function esc(value) {{
      return textOf(value).replace(/[&<>"']/g, (c) => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
    }}

    function searchable(example) {{
      const token = example.token_info || {{}};
      return [
        example.record_id, example.prompt, example.media_type, example.media_path,
        example.tags, token.kind, token.phase, token.role, token.token_text, token.text_context, token.visual_position
      ].map(textOf).join(" ").toLowerCase();
    }}

    function renderBreakdown(rows) {{
      return rows.map((row) => `<span class="pill">${{esc(row.label)}} ${{esc(row.count)}}</span>`).join("");
    }}

    function tokenDetail(example) {{
      const token = example.token_info || {{}};
      const pos = token.visual_position || {{}};
      if (token.kind === "text") {{
        return `${{esc(token.token_text || "")}}${{token.text_context ? `<div class="path">${{esc(token.text_context)}}</div>` : ""}}`;
      }}
      if (token.visual_position) {{
        return `frame ${{esc(pos.frame)}} - patch (${{esc(pos.patch_x)}}, ${{esc(pos.patch_y)}})`;
      }}
      return esc(token.token_text || "");
    }}

    function renderExample(example, cssClass, score) {{
      const token = example.token_info || {{}};
      const tags = Array.isArray(example.tags) ? example.tags : [];
      const scoreHtml = score == null ? "" : `<span class="pill score">sim ${{Number(score).toFixed(4)}}</span>`;
      return `
        <article class="${{cssClass}}">
          <strong>${{esc(example.record_id || "(unknown record)")}}</strong>
          <div class="meta">
            ${{scoreHtml}}
	            <span class="pill">${{esc(example.media_type || "unknown")}}</span>
	            <span class="pill">token ${{esc(example.token_index)}}</span>
	            ${{token.kind ? `<span class="pill">${{esc(token.kind)}}</span>` : ""}}
	            ${{token.phase ? `<span class="pill">${{esc(token.phase)}}</span>` : ""}}
	            ${{token.role ? `<span class="pill">${{esc(token.role)}}</span>` : ""}}
	            <span class="pill">${{esc(example.shard || "")}}</span>
            ${{tags.map((tag) => `<span class="pill">${{esc(tag)}}</span>`).join("")}}
          </div>
          <div class="prompt">${{esc(example.prompt || "")}}</div>
          ${{token.kind ? `<div class="path">${{tokenDetail(example)}}</div>` : ""}}
          ${{example.media_path ? `<div class="path">${{esc(example.media_path)}}</div>` : ""}}
        </article>
      `;
    }}

    function render() {{
      const q = searchEl.value.trim().toLowerCase();
      const visible = ROWS.filter((row) => !q || searchable(row.query).includes(q) || row.neighbors.some((n) => searchable(n).includes(q)));
      const summaryHtml = `
        <section class="summary">
          <div class="metric">
            <div class="label">Queries</div>
            <div class="value">${{esc(SUMMARY.query_count || 0)}}</div>
            <div class="breakdown">${{renderBreakdown(SUMMARY.query_breakdown || [])}}</div>
          </div>
          <div class="metric">
            <div class="label">Neighbors</div>
            <div class="value">${{esc(SUMMARY.neighbor_count || 0)}}</div>
            <div class="breakdown">${{renderBreakdown(SUMMARY.neighbor_breakdown || [])}}</div>
          </div>
          <div class="metric">
            <div class="label">Visible query groups</div>
            <div class="value">${{esc(visible.length)}}</div>
          </div>
        </section>
      `;
      contentEl.innerHTML = summaryHtml + (visible.map((row, idx) => `
        <section class="group">
          <div class="query">
            <div class="meta"><span class="pill">query ${{idx + 1}}</span></div>
            ${{renderExample(row.query, "query-inner", null)}}
          </div>
          <div class="neighbors">
            ${{row.neighbors.map((neighbor) => renderExample(neighbor, "neighbor", neighbor.similarity)).join("")}}
          </div>
        </section>
      `).join("") || "<p>No matching neighbors.</p>");
    }}

    searchEl.addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""
