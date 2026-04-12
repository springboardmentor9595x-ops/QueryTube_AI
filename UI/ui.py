import gradio as gr
import requests
import pandas as pd

from search_engine import (
    returnSearchResults,
    DEFAULT_TOPK,
    DEFAULT_METRIC,
    DEFAULT_TW,
)

# Load datetime info once for sorting
_meta = pd.read_csv("cleaned_transcripts.csv")[["video_id", "datetime"]]
_meta["datetime"] = pd.to_datetime(_meta["datetime"], utc=True, errors="coerce")
DATE_MAP = dict(zip(_meta["video_id"], _meta["datetime"]))


# ─────────────────────────────────────────────────────
# Thumbnail helper
# ─────────────────────────────────────────────────────

def get_thumbnail(video_id: str) -> str:
    for quality in ["sddefault", "hqdefault", "mqdefault", "default"]:
        url = f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"
        try:
            if requests.head(url, timeout=3).status_code == 200:
                return url
        except Exception:
            continue
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


# ─────────────────────────────────────────────────────
# Sort helper
# ─────────────────────────────────────────────────────

def sort_results(results: list, sort_by: str) -> list:
    if sort_by == "Relevance (default)":
        return results
    elif sort_by == "Title A → Z":
        return sorted(results, key=lambda r: r["title"].lower())
    elif sort_by == "Newest first":
        return sorted(
            results,
            key=lambda r: DATE_MAP.get(r["video_id"], pd.Timestamp.min.tz_localize("UTC")),
            reverse=True,
        )
    elif sort_by == "Oldest first":
        return sorted(
            results,
            key=lambda r: DATE_MAP.get(r["video_id"], pd.Timestamp.max.tz_localize("UTC")),
        )
    return results


# ─────────────────────────────────────────────────────
# Build HTML result cards
# ─────────────────────────────────────────────────────

def _build_cards(results: list, query: str, threshold: float, sort_by: str) -> str:
    if not results:
        return (
            "<div style='text-align:center; margin-top:40px;'>"
            "<h3>No results found above the threshold.</h3>"
            "</div>"
        )

    results = sort_results(results, sort_by)

    cards_html = ""
    for i, r in enumerate(results, 1):
        score_pct = min(max(r["score"], 0), 1)
        bar_color = f"hsl({int(score_pct * 120)}, 65%, 42%)"
        bar_width = f"{int(score_pct * 100)}%"
        thumb = get_thumbnail(r["video_id"])

        dt = DATE_MAP.get(r["video_id"])
        date_str = (
            f"<span style='font-size:12px; color:#888;'>📅 {dt.strftime('%b %d, %Y')}</span>"
            if pd.notna(dt) else ""
        )

        cards_html += f"""
        <div style="display:flex; gap:16px; border:1px solid #ddd;
                    padding:14px; border-radius:10px; margin-bottom:14px;">
          <div style="min-width:32px; height:32px; background:#6c47d4;
                      border-radius:50%; display:flex; align-items:center;
                      justify-content:center; color:white;">{i}</div>

          <a href="{r['youtube_link']}" target="_blank">
            <img src="{thumb}" style="width:160px; height:90px;
                 border-radius:8px;" />
          </a>

          <div style="flex:1;">
            <h3 style="margin:0;">{r['title']}</h3>
            {date_str}

            <div style="margin-top:6px;">
              <div style="font-size:13px;">Score: {r['score']:.4f}</div>
              <div style="background:#ddd; height:6px; border-radius:4px;">
                <div style="width:{bar_width}; background:{bar_color};
                            height:6px; border-radius:4px;"></div>
              </div>
            </div>

            <a href="{r['youtube_link']}" target="_blank"
               style="display:inline-block; margin-top:8px;
                      padding:5px 12px; background:red; color:white;
                      border-radius:5px;">▶ Watch</a>
          </div>
        </div>
        """

    header = f"""
    <div style="margin-bottom:12px;">
        Showing <b>{len(results)}</b> results for <b>"{query}"</b>
    </div>
    """

    return header + cards_html


# ─────────────────────────────────────────────────────
# Search handler
# ─────────────────────────────────────────────────────

def run_search(query, top_k, threshold, sort_by, history):
    if not query.strip():
        return "Enter query", history, gr.Radio(choices=list(reversed(history)))

    if not history or history[-1] != query.strip():
        history = history + [query.strip()]

    results = returnSearchResults(
        query,
        metric=DEFAULT_METRIC,
        title_weight=DEFAULT_TW,
        top_k=int(top_k),
        threshold=float(threshold),
    )

    html = _build_cards(results, query, threshold, sort_by)
    return html, history, gr.Radio(choices=list(reversed(history)))


def resort(query, top_k, threshold, sort_by, history):
    return run_search(query, top_k, threshold, sort_by, history)


# ─────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────

with gr.Blocks(title="QueryTube") as demo:

    history_state = gr.State([])
    sidebar_visible = gr.State(False)
    about_visible = gr.State(False)   # ✅ NEW

    # Sidebar CSS
    gr.HTML("""
    <style>
    #sidebar {
        position: fixed;
        left: 0;
        top: 0;
        width: 360px;
        height: 100%;
        background: white;
        border-right: 1px solid #ddd;
        padding: 15px;
        z-index: 1000;
    }
    </style>
    """)

    # Header
    gr.HTML("""
    <div style="text-align:center; margin-top:10px;">
        <h1 style="margin-bottom:5px;">🔍 QueryTube</h1>
        <h2 style="color:red; margin:5px;">Your Semantic YouTube Search Engine</h2>
        <h3 style="margin-top:5px;">Search whatever you want</h3>
    </div>
    """)

    # Sidebar
    sidebar = gr.Column(visible=False, elem_id="sidebar")

    with sidebar:
        topk_dd = gr.Dropdown([3,5,7,10], value=DEFAULT_TOPK, label="Top-K")
        threshold_slider = gr.Slider(0,1,0.15,step=0.01,label="Threshold")
        history_box = gr.Radio(choices=[], label="History")

    # Search + sort
    with gr.Row():
        query_box = gr.Textbox(label="Search", scale=5)

        with gr.Column(scale=1, min_width=140):
            search_btn = gr.Button("Search 🔍")
            sort_dd = gr.Dropdown(
                ["Relevance (default)", "Title A → Z", "Newest first", "Oldest first"],
                value="Relevance (default)",
                label="Sort"
            )

    results_html = gr.HTML("Results appear here")

    # Settings button
    with gr.Row():
        sidebar_btn = gr.Button("⚙️ Settings")

    # 📌 About button
    about_btn = gr.Button("📌 About")

    # About section (hidden initially)
    about_section = gr.HTML("""
    <hr style="margin-top:40px;">
    <div style="padding:10px;">
        <h3>📌 About</h3>
        <ul>
            <li>This project is a semantic search engine for YouTube videos using embeddings.</li>
            <li>It retrieves relevant videos based on meaning, not just keyword matching.</li>
            <li>Users can filter results using Top-K and relevance threshold.</li>
            <li>Sorting options include relevance, alphabetical order, and upload date.</li>
            <li>The system enhances search experience with visual scores and video previews.</li>
        </ul>
    </div>
    """, visible=False)

    # ── Events ──

    def toggle_sidebar(vis):
        return not vis, gr.update(visible=not vis)

    def toggle_about(vis):
        return not vis, gr.update(visible=not vis)

    sidebar_btn.click(toggle_sidebar, inputs=[sidebar_visible], outputs=[sidebar_visible, sidebar])

    about_btn.click(toggle_about, inputs=[about_visible], outputs=[about_visible, about_section])

    inputs = [query_box, topk_dd, threshold_slider, sort_dd, history_state]
    outputs = [results_html, history_state, history_box]

    search_btn.click(run_search, inputs, outputs)
    query_box.submit(run_search, inputs, outputs)
    sort_dd.change(resort, inputs, outputs)

    history_box.change(
        fn=lambda x: x if x else "",
        inputs=[history_box],
        outputs=[query_box],
    )


if __name__ == "__main__":
    demo.launch()