import pandas as pd
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.5
TITLE_WEIGHT = 0.3
TRANSCRIPT_WEIGHT = 0.7

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("cleaned_transcripts.csv")
df["title"] = df["title"].fillna("").astype(str)
df["transcript"] = df["transcript"].fillna("").astype(str)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
model = SentenceTransformer(MODEL_NAME)

# -------------------------------------------------
# PRECOMPUTE EMBEDDINGS
# -------------------------------------------------
title_embeddings = model.encode(df["title"].tolist(), show_progress_bar=True)
transcript_embeddings = model.encode(df["transcript"].tolist(), show_progress_bar=True)

# -------------------------------------------------
# SEARCH FUNCTION
# -------------------------------------------------
def search_videos(query, top_k, threshold):
    if not query.strip():
        return """
        <div style="color:#f87171; font-size:24px; font-weight:700; text-align:center; padding:30px;">
            Please enter a search query.
        </div>
        """

    query_embedding = model.encode([query])[0].reshape(1, -1)

    title_scores = cosine_similarity(query_embedding, title_embeddings)[0]
    transcript_scores = cosine_similarity(query_embedding, transcript_embeddings)[0]

    final_scores = (TITLE_WEIGHT * title_scores) + (TRANSCRIPT_WEIGHT * transcript_scores)

    results_df = df.copy()
    results_df["final_score"] = final_scores

    filtered_df = results_df[results_df["final_score"] >= threshold]
    filtered_df = filtered_df.sort_values(by="final_score", ascending=False).head(int(top_k))

    if filtered_df.empty:
        return """
        <div style="color:#f87171; font-size:24px; font-weight:700; text-align:center; padding:30px;">
            No relevant videos found.
        </div>
        """

    html_output = """
    <style>
        .cards-container {
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            justify-content: flex-start;
        }

        .video-card {
            width: 410px;
            background: #0f172a;
            border-radius: 22px;
            padding: 18px;
            border: 1px solid #94a3b8;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }

        .video-title {
            font-size: 24px;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.35;
            margin-bottom: 12px;
        }

        .video-score {
            font-size: 18px;
            color: #f8fafc;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .video-desc {
            font-size: 16px;
            color: #e2e8f0;
            line-height: 1.6;
            margin-bottom: 14px;
        }

        .video-link a {
            color: #38bdf8;
            font-size: 18px;
            font-weight: 700;
            text-decoration: underline;
        }

        .video-link a:hover {
            color: #7dd3fc;
        }

        iframe {
            width: 100%;
            height: 230px;
            border: none;
            border-radius: 16px;
            margin-top: 14px;
        }
    </style>

    <div class="cards-container">
    """

    for _, row in filtered_df.iterrows():
        title = row["title"]
        video_id = row["video_id"]
        score = row["final_score"]
        short_desc = row["transcript"][:180] + "..." if len(row["transcript"]) > 180 else row["transcript"]

        youtube_link = f"https://www.youtube.com/watch?v={video_id}"
        embed_link = f"https://www.youtube.com/embed/{video_id}"

        html_output += f"""
        <div class="video-card">
            <div class="video-title">{title}</div>
            <div class="video-score">Relevance Score: {score:.3f}</div>
            <div class="video-desc">{short_desc}</div>
            <div class="video-link">
                <a href="{youtube_link}" target="_blank">Watch on YouTube</a>
            </div>
            <iframe src="{embed_link}" allowfullscreen></iframe>
        </div>
        """

    html_output += "</div>"
    return html_output

# -------------------------------------------------
# SETTINGS VISIBILITY
# -------------------------------------------------
def open_settings():
    return gr.update(visible=True)

def close_settings():
    return gr.update(visible=False)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
custom_css = """
body {
    background: #020617 !important;
}

.gradio-container {
    max-width: 100% !important;
    background: #020617 !important;
}

footer {
    visibility: hidden;
}

#main-title-wrap {
    text-align: center;
    margin-top: 20px;
    margin-bottom: 28px;
}

#main-title-wrap h1 {
    color: white;
    font-size: 64px;
    font-weight: 900;
    margin-bottom: 10px;
}

#main-title-wrap h3 {
    color: #ffffff;
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 12px;
}

#main-title-wrap p {
    color: #f8fafc;
    font-size: 18px;
}

/* Left search card */
#search-panel {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}

#query-box label {
    color: white !important;
    font-weight: 800 !important;
    font-size: 18px !important;
}

#query-box textarea {
    background: #374151 !important;
    color: #ffffff !important;
    border: 1px solid #4b5563 !important;
    border-radius: 14px !important;
    font-size: 20px !important;
    min-height: 90px !important;
}

/* Buttons */
#settings-btn button {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    min-height: 74px !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

#search-btn button {
    background: linear-gradient(90deg, #4f46e5 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    min-height: 74px !important;
    font-size: 22px !important;
    font-weight: 800 !important;
}

/* Settings box under buttons */
#settings-box {
    margin-top: 16px !important;
    background: #111827 !important;
    border: 1px solid #334155 !important;
    border-radius: 18px !important;
    padding: 18px !important;
}

#settings-box h2 {
    color: white !important;
    font-size: 24px !important;
    margin-bottom: 16px !important;
}

#settings-box label {
    color: white !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

#close-settings button {
    background: #ef4444 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 52px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    margin-top: 12px !important;
}

input[type="range"] {
    accent-color: #6366f1 !important;
}

#results-output {
    min-height: 500px;
}
"""

# -------------------------------------------------
# UI
# -------------------------------------------------
with gr.Blocks(css=custom_css, theme=gr.themes.Base(), title="QueryTube AI") as demo:
    gr.HTML("""
    <div id="main-title-wrap">
        <h1>QueryTube AI</h1>
        <h3>A Semantic Video Search platform</h3>
        <p>Enter a query and get the most relevant YouTube videos.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_id="search-panel"):
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="what is ai agents?",
                lines=2,
                elem_id="query-box"
            )

            with gr.Row():
                settings_btn = gr.Button("⚙️ Settings", elem_id="settings-btn")
                search_btn = gr.Button("Search", elem_id="search-btn")

            with gr.Column(visible=False, elem_id="settings-box") as settings_box:
                gr.HTML("<h2>Settings</h2>")

                top_k_input = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=DEFAULT_TOP_K,
                    step=1,
                    label="Top K Results"
                )

                threshold_input = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    value=DEFAULT_THRESHOLD,
                    step=0.05,
                    label="Threshold"
                )

                close_btn = gr.Button("Close", elem_id="close-settings")

        with gr.Column(scale=2):
            results_output = gr.HTML(elem_id="results-output")

    settings_btn.click(fn=open_settings, outputs=settings_box)
    close_btn.click(fn=close_settings, outputs=settings_box)

    search_btn.click(
        fn=search_videos,
        inputs=[query_input, top_k_input, threshold_input],
        outputs=results_output
    )

demo.launch()