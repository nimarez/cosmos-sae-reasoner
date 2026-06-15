from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .artifacts import ensure_dir, read_jsonl


def render_feature_report(features_path: Path, output: Path, *, title: str = "Cosmos SAE Feature Browser") -> None:
    rows = read_jsonl(features_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("feature_id", "unknown"))].append(row)
    features = []
    for feature_id, examples in sorted(grouped.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        examples.sort(key=lambda row: float(row.get("activation", 0.0)), reverse=True)
        max_activation = max((float(row.get("activation", 0.0)) for row in examples), default=0.0)
        features.append({"feature_id": feature_id, "max_activation": max_activation, "examples": examples})
    ensure_dir(output.parent)
    output.write_text(render_html(features, title=title), encoding="utf-8")


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
          <div class="metric"><div class="label">Max activation</div><div class="value">${{Number(feature.max_activation || 0).toFixed(3)}}</div></div>
        </div>
        <div class="examples">
          ${{feature.examples.map((row) => renderExample(row, max)).join("")}}
        </div>
      `;
    }}

    function renderExample(row, max) {{
      const activation = Number(row.activation || 0);
      const width = Math.max(2, Math.min(100, 100 * activation / max));
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
