# -----------------------------
# QUERYTUBE AI - CLEAN UI
# -----------------------------

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# -----------------------------
# LOAD DATA
# -----------------------------

video_index = pd.read_csv("../vid_index/video_index.csv")

embedding_cols = [col for col in video_index.columns if "embedding_" in col]
embeddings = video_index[embedding_cols].values

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# HELPERS
# -----------------------------

def _empty_state(msg):
    return f"""
    <div style='text-align:center; padding:48px 0; color:#9ca3af; font-size:15px;'>
      {msg}
    </div>
    """

def _build_top3_html():
    """Return HTML for the top-3 most popular videos."""
    pop_col = None
    for col in ['view_count', 'views', 'viewCount', 'likes', 'like_count']:
        if col in video_index.columns:
            pop_col = col
            break

    if pop_col:
        top3 = video_index.nlargest(3, pop_col)
    else:
        date_col = None
        for col in ['published_at', 'date', 'published', 'upload_date', 'publishedAt']:
            if col in video_index.columns:
                date_col = col
                break
        if date_col:
            top3 = video_index.copy()
            top3[date_col] = pd.to_datetime(top3[date_col], errors='coerce')
            top3 = top3.nlargest(3, date_col)
        else:
            top3 = video_index.head(3)

    cards = ""
    crowns = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(top3.iterrows()):
        video_id = row['video_id']
        title    = row['title']
        link     = f"https://www.youtube.com/watch?v={video_id}"
        thumb    = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        stat_html = ""
        if pop_col:
            val_raw = row.get(pop_col, None)
            if val_raw is not None and not (isinstance(val_raw, float) and val_raw != val_raw):
                val = int(val_raw)
                formatted = f"{val:,}" if val < 1_000_000 else f"{val/1_000_000:.1f}M"
                icon = "👁" if "view" in pop_col.lower() else "👍"
                stat_html = f'<div style="font-size:11px; color:#a3e635; font-weight:600; margin-top:6px;">{icon} {formatted}</div>'

        cards += f"""
        <a href="{link}" target="_blank" style="text-decoration:none; display:block; flex:1; min-width:0;">
          <div style="display:flex; flex-direction:column; border-radius:10px;
                      background:rgba(255,255,255,0.07); border:1.5px solid rgba(255,255,255,0.15);
                      overflow:hidden; height:100%;"
               onmouseover="this.style.background='rgba(255,255,255,0.13)'; this.style.transform='translateY(-3px)'"
               onmouseout="this.style.background='rgba(255,255,255,0.07)'; this.style.transform='none'">
            <div style="position:relative;">
              <img src="{thumb}" style="width:100%; height:130px; object-fit:cover; display:block;" />
              <div style="position:absolute; top:7px; left:7px; font-size:18px; line-height:1;">{crowns[i]}</div>
            </div>
            <div style="padding:10px 12px 12px;">
              <div style="font-size:13px; font-weight:700; color:#fef9c3; line-height:1.35;
                          display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">
                {title}
              </div>
              {stat_html}
            </div>
          </div>
        </a>
        """

    return f"""
    <div style="margin-bottom:6px;">
      <div style="font-size:12px; font-weight:700; color:#a3e635; letter-spacing:0.08em;
                  text-transform:uppercase; margin-bottom:10px;">
        &#9733; Top Videos on this Channel
      </div>
      <div style="display:flex; gap:12px; align-items:stretch;">
        {cards}
      </div>
    </div>
    """

TOP3_HTML = _build_top3_html()

# -----------------------------
# SEARCH FUNCTION
# -----------------------------

def search_videos(query, top_k, threshold, sort_by):
    if not query.strip():
        return _empty_state("Type something above to search ✨"), ""

    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]
    filtered_indices = [i for i in top_indices if scores[i] >= threshold]

    if len(filtered_indices) == 0:
        filtered_indices = list(top_indices)

    if sort_by == "Highest Score":
        filtered_indices = sorted(filtered_indices, key=lambda i: scores[i], reverse=True)
    elif sort_by == "Lowest Score":
        filtered_indices = sorted(filtered_indices, key=lambda i: scores[i])

    count = len(filtered_indices)
    count_text = f"### 🎯 {count} result{'s' if count != 1 else ''} found for \"{query}\""

    cards = ""
    for rank, idx in enumerate(filtered_indices, 1):
        row       = video_index.iloc[idx]
        video_id  = row['video_id']
        title     = row['title']
        score     = float(scores[idx])
        score_pct = round(score * 100, 1)

        # Try common date column names, fall back gracefully
        pub_raw = None
        for col in ['published_at', 'date', 'published', 'upload_date', 'publishedAt']:
            if col in row and not pd.isna(row[col]):
                pub_raw = row[col]
                break

        if pub_raw:
            try:
                dt = pd.to_datetime(pub_raw)
                date_str = dt.strftime("%b %d, %Y")
                time_str = dt.strftime("%I:%M %p")
            except Exception:
                date_str = str(pub_raw)[:10]
                time_str = ""
        else:
            date_str, time_str = None, None

        if score >= 0.6:
            badge_bg, badge_fg, label = "#dcfce7", "#14532d", "&#10003; Strong Match"
        elif score >= 0.4:
            badge_bg, badge_fg, label = "#fef3c7", "#92400e", "Good Match"
        else:
            badge_bg, badge_fg, label = "#ffedd5", "#9a3412", "Possible Match"

        link  = f"https://www.youtube.com/watch?v={video_id}"
        thumb = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        date_row = ""
        if date_str:
            time_part = f" &nbsp;&nbsp; {time_str}" if time_str else ""
            date_row = f'<div style="font-size:12px; color:#6b7280; margin-bottom:4px;">&#128197; {date_str}{time_part}</div>'

        cards += f"""
        <a href="{link}" target="_blank" style="text-decoration:none; display:block;">
          <div style="display:flex; flex-direction:column; border-radius:12px;
                      background:#ffffff; border:1.5px solid #e5e7eb; overflow:hidden;"
               onmouseover="this.style.boxShadow='0 4px 16px rgba(180,83,9,0.15)'; this.style.transform='translateY(-3px)'"
               onmouseout="this.style.boxShadow='none'; this.style.transform='none'">

            <div style="position:relative;">
              <img src="{thumb}" style="width:100%; height:160px; object-fit:cover; display:block;" />
              <div style="position:absolute; top:8px; left:8px; background:rgba(0,0,0,0.65);
                          color:#fff; font-size:11px; font-weight:700; padding:2px 7px; border-radius:4px;">
                #{rank}
              </div>
            </div>

            <div style="padding:12px 14px 14px;">

              <div style="font-size:14px; font-weight:700; color:#111827; line-height:1.4; margin-bottom:8px;
                          display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;">
                {title}
              </div>

              <div style="font-size:12px; color:#6b7280; margin-bottom:4px;">
                &#128228; <span style="font-family:monospace; color:#374151; font-size:11px;">{video_id}</span>
              </div>

              {date_row}

              <div style="display:flex; align-items:center; justify-content:space-between; margin-top:8px;">
                <span style="background:{badge_bg}; color:{badge_fg};
                              font-size:11px; font-weight:700; padding:3px 9px; border-radius:20px;">
                  {label}
                </span>
                <span style="font-size:13px; font-weight:700; color:#b45309;">
                  {score_pct}% match
                </span>
              </div>

            </div>
          </div>
        </a>
        """

    grid = f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; padding:2px;">
      {cards}
    </div>
    """
    return grid, count_text


def clear_all():
    return "", _empty_state("Type something above to search ✨"), ""


# -----------------------------
# UI
# -----------------------------

OLIVE_CSS = """
body, .gradio-container { background: #2d3b1e !important; }
.gradio-container { background: #2d3b1e !important; }
footer { display:none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="orange",
        secondary_hue="green",
        neutral_hue="stone",
    ),
    css=OLIVE_CSS,
    title="QueryTubeAI"
) as demo:

    # Header
    gr.HTML("""
    <div style="text-align:center; padding:28px 0 16px;">
      <div style="margin-bottom:4px;">
        <span style="
          font-size:62px;
          font-weight:900;
          letter-spacing:-3px;
          line-height:1;
          color:#f59e0b;
          text-shadow: 0 2px 0 #92400e, 0 4px 16px rgba(245,158,11,0.35);
          display:inline;
        ">QueryTube</span><span style="
          font-size:62px;
          font-weight:300;
          letter-spacing:-2px;
          line-height:1;
          color:#a3e635;
          text-shadow: 0 2px 0 #365314, 0 4px 16px rgba(163,230,53,0.3);
          display:inline;
        ">AI</span>
      </div>
      <p style="color:#78716c; font-size:15px; margin:8px 0 16px;">
        Search videos by <em>meaning</em>, not keywords
      </p>
      <div style="display:inline-flex; gap:10px; flex-wrap:wrap; justify-content:center;">
        <span style="background:#fef3c7; color:#92400e; font-size:13px; font-weight:600; padding:4px 14px; border-radius:99px;">🧠 Semantic Search</span>
        <span style="background:#dcfce7; color:#14532d; font-size:13px; font-weight:600; padding:4px 14px; border-radius:99px;">⚡ Instant Results</span>
        <span style="background:#ffedd5; color:#9a3412; font-size:13px; font-weight:600; padding:4px 14px; border-radius:99px;">🎯 Cosine Similarity</span>
      </div>
    </div>
    """)

    # Top 3 featured videos
    gr.HTML(TOP3_HTML)

    # Divider
    gr.HTML("""
    <div style="border-top:1px solid rgba(255,255,255,0.12); margin:8px 0 16px;"></div>
    """)

    # Search bar
    with gr.Row(equal_height=True):
        query_input = gr.Textbox(
            placeholder="e.g. 'how to get a developer job'  or  'machine learning basics'",
            show_label=False,
            scale=5,
        )
        search_btn = gr.Button("🔍  Search", variant="primary", scale=1, min_width=110)
        clear_btn  = gr.Button("✕  Clear",  variant="secondary", scale=1, min_width=90)

    # Result count label
    result_label = gr.Markdown("")

    # Results
    results_html = gr.HTML(_empty_state("Type something above to search ✨"))

    # Settings accordion
    with gr.Accordion("⚙️  Search Settings", open=False):
        with gr.Row():
            top_k_slider     = gr.Slider(1, 20, value=6, step=1,    label="Max Results")
            threshold_slider = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Min Similarity Threshold")
            sort_dropdown    = gr.Dropdown(
                choices=["Highest Score", "Lowest Score", "Default"],
                value="Highest Score",
                label="Sort By"
            )

    # How it works
    with gr.Accordion("ℹ️  How it works", open=False):
        gr.HTML("""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:4px;">
          <div style="background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:14px;">
            <strong style="color:#92400e; font-size:14px;">🧩 Model</strong><br>
            <span style="color:#78716c; font-size:13px;">all-MiniLM-L6-v2 — 384-dim sentence embeddings.</span>
          </div>
          <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px;">
            <strong style="color:#14532d; font-size:14px;">📐 Ranking</strong><br>
            <span style="color:#78716c; font-size:13px;">Cosine similarity between query and video embeddings.</span>
          </div>
          <div style="background:#fff7ed; border:1px solid #fed7aa; border-radius:10px; padding:14px;">
            <strong style="color:#9a3412; font-size:14px;">🎨 Match Labels</strong><br>
            <span style="color:#78716c; font-size:13px;">Strong ≥60% · Good ≥40% · Possible &lt;40%</span>
          </div>
          <div style="background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:14px;">
            <strong style="color:#92400e; font-size:14px;">⌨️ Tip</strong><br>
            <span style="color:#78716c; font-size:13px;">Press <kbd>Enter</kbd> in the search box to search instantly.</span>
          </div>
        </div>
        """)

    # Events
    search_btn.click(
        fn=search_videos,
        inputs=[query_input, top_k_slider, threshold_slider, sort_dropdown],
        outputs=[results_html, result_label]
    )
    query_input.submit(
        fn=search_videos,
        inputs=[query_input, top_k_slider, threshold_slider, sort_dropdown],
        outputs=[results_html, result_label]
    )
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[query_input, results_html, result_label]
    )

# -----------------------------
# LAUNCH
# -----------------------------

demo.launch()