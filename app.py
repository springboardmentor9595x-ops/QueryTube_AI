import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="QueryTube",
    page_icon="🎬",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #11111b; }
    .stApp { background-color: #11111b; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stButton > button {
        background: #7c3aed !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        width: 100% !important;
        height: 45px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
            
    .stTextInput > div > div > input {
        background: #1e1e2e !important;
        border: 1px solid #313244 !important;
        color: #cdd6f4 !important;
        border-radius: 10px !important;
        font-size: 15px !important;
    }
            
    .stTextInput > div:focus-within {
        border-color: white !important;
        box-shadow: none !important;
    }

    [data-testid="stThumbValue"], 
    div[data-baseweb="slider"] span,
    div[data-baseweb="slider"] div {
        color: white !important;
        -webkit-text-fill-color: white !important;
    }

    div[role="slider"] {
        background-color: white !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[role="slider"]:focus, div[role="slider"]:active {
        box-shadow: 0 0 0 2px white !important;
    }

    div[data-baseweb="slider"] > div > div > div {
        background-color: white !important;
        background: white !important;
    }

    .stMarkdown, .stSlider label { 
        color: #cdd6f4 !important; 
    }
</style>
""", unsafe_allow_html=True)

# ── Configuration ────────────────────────────────────────────
MODEL_NAME = 'all-mpnet-base-v2'
TOP_K      = 5
THRESHOLD  = 0.3

# ── Load Data and Model ──────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data():
    df        = pd.read_csv('video_index.csv')
    meta_cols = ['video_id', 'title', 'datetime', 'transcript']
    emb_cols  = [col for col in df.columns if col.startswith('emb_')]
    meta       = df[meta_cols].copy()
    embeddings = df[emb_cols].values
    return meta, embeddings

model            = load_model()
meta, embeddings = load_data()

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:36px 0 24px 0;">
    <div style="font-size:2.8rem;">🎬</div>
    <h1 style="color:#cdd6f4; font-size:2.4rem; margin:0 0 8px 0;">QueryTube</h1>
    <p style="color:#6c7086; font-size:0.95rem; margin:0 0 14px 0;">
        Semantic search for PBS Be Smart by Joe Hanson — powered by all-mpnet-base-v2
    </p>
    <div style="display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">
        <span style="background:#181825; border:1px solid #313244; border-radius:999px; padding:4px 14px; font-size:0.75rem; color:#a6e3a1;">✅ 123 videos indexed</span>
        <span style="background:#181825; border:1px solid #313244; border-radius:999px; padding:4px 14px; font-size:0.75rem; color:#89b4fa;">🧠 all-mpnet-base-v2</span>
        <span style="background:#181825; border:1px solid #313244; border-radius:999px; padding:4px 14px; font-size:0.75rem; color:#f9e2af;">📊 98.8% Top-3 Recall</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Search Bar ───────────────────────────────────────────────
col1, col2 = st.columns([9, 1])
with col1:
    query = st.text_input(
        "",
        placeholder="Ask anything... e.g. What is dust? How do birds fly?",
        label_visibility="collapsed"
    )
with col2:
    search_clicked = st.button("🔍 Search")

# ── Filters ──────────────────────────────────────────────────
col3, col4 = st.columns(2)
with col3:
    top_k = st.slider("Top-K Results", min_value=1, max_value=10, value=TOP_K, step=1)
with col4:
    threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=THRESHOLD, step=0.05)

# ── Search Logic ─────────────────────────────────────────────
if search_clicked or query:
    if not query.strip():
        st.markdown("<p style='color:#6c7086; text-align:center;'>Please enter a search query.</p>", unsafe_allow_html=True)
    else:
        query_embedding = model.encode([query])
        scores          = cosine_similarity(query_embedding, embeddings)[0]
        ranked_all      = np.argsort(scores)[::-1]
        mask            = scores[ranked_all] >= threshold
        filtered        = ranked_all[mask][:int(top_k)]

        if len(filtered) == 0:
            st.markdown(f"""
            <div style='text-align:center; padding:60px 20px;'>
                <div style='font-size:2.5rem;'>🔍</div>
                <p style='color:#ff6b6b;'>No results found above threshold {threshold}.</p>
                <p style='color:#6c7086; font-size:0.85rem;'>Try lowering the threshold or using different keywords.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            avg_score = round(float(np.mean([scores[i] for i in filtered])), 3)
            top_score = round(float(scores[filtered[0]]), 3)

            # ── Stats Bar ─────────────────────────────────────
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg, #1e1e2e, #181825);
                border:1px solid #313244;
                border-radius:14px;
                padding:16px 24px;
                margin-bottom:24px;
                display:flex;
                gap:32px;
                flex-wrap:wrap;
            ">
                <div>
                    <div style='font-size:11px; color:#6c7086; margin-bottom:4px;'>QUERY</div>
                    <div style='font-size:14px; color:#cdd6f4; font-weight:600;'>"{query}"</div>
                </div>
                <div>
                    <div style='font-size:11px; color:#6c7086; margin-bottom:4px;'>RESULTS</div>
                    <div style='font-size:14px; color:#a6e3a1; font-weight:600;'>{len(filtered)} found</div>
                </div>
                <div>
                    <div style='font-size:11px; color:#6c7086; margin-bottom:4px;'>TOP SCORE</div>
                    <div style='font-size:14px; color:#89b4fa; font-weight:600;'>{top_score}</div>
                </div>
                <div>
                    <div style='font-size:11px; color:#6c7086; margin-bottom:4px;'>AVG SCORE</div>
                    <div style='font-size:14px; color:#fab387; font-weight:600;'>{avg_score}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Results Cards ─────────────────────────────────
            for rank, idx in enumerate(filtered, 1):
                video_id   = meta.iloc[idx]['video_id']
                title      = meta.iloc[idx]['title']
                date       = meta.iloc[idx]['datetime']
                score      = round(float(scores[idx]), 3)
                transcript = str(meta.iloc[idx]['transcript'])[:200] + "..."
                link       = f"https://www.youtube.com/watch?v={video_id}"
                embed_url  = f"https://www.youtube.com/embed/{video_id}"
                thumb      = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
                bar        = int(score * 100)

                if score >= 0.5:
                    color    = "#a6e3a1"
                    badge_bg = "#1a3a2a"
                    badge    = "Strong Match"
                elif score >= 0.3:
                    color    = "#f9e2af"
                    badge_bg = "#3a3020"
                    badge    = "Good Match"
                else:
                    color    = "#BE2323"
                    badge_bg = "#3a2020"
                    badge    = "Weak Match"

                # ✅ Poora card ek components.html mein
                components.html(f"""
                <!DOCTYPE html>
                <html>
                <head>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{ background: transparent; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
                </style>
                </head>
                <body>
                <div style="
                    background:#1e1e2e;
                    border:1px solid #313244;
                    border-radius:16px;
                    margin-bottom:4px;
                    overflow:hidden;
                ">
                    <!-- Top: Info + Thumbnail -->
                    <div style="display:flex; flex-wrap:wrap;">

                        <!-- Info -->
                        <div style="flex:1; min-width:260px; padding:20px 24px;">
                            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                                <span style="background:#313244; color:#6c7086; font-size:11px; font-weight:700; padding:3px 10px; border-radius:999px;">#{rank}</span>
                                <span style="background:{badge_bg}; color:{color}; font-size:11px; font-weight:600; padding:3px 10px; border-radius:999px;">{badge}</span>
                            </div>

                            <a href="{link}" target="_blank" style="font-size:16px; font-weight:700; color:#cdd6f4; text-decoration:none; display:block; margin-bottom:8px;">{title}</a>

                            <div style="display:flex; gap:16px; margin-bottom:14px;">
                                <span style="font-size:12px; color:#6c7086;">📅 {date}</span>
                                <span style="font-size:12px; color:#6c7086;">📺 PBS Be Smart</span>
                            </div>

                            <div style="background:#181825; border-left:3px solid #313244; border-radius:0 8px 8px 0; padding:10px 14px; margin-bottom:16px;">
                                <div style="font-size:10px; color:#585b70; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.5px;">Transcript Preview</div>
                                <div style="font-size:12px; color:#9399b2; line-height:1.6;">{transcript}</div>
                            </div>

                            <div style="margin-bottom:16px;">
                                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                                    <span style="font-size:11px; color:#6c7086;">Similarity Score</span>
                                    <span style="font-size:12px; color:{color}; font-weight:700;">{score}</span>
                                </div>
                                <div style="background:#313244; border-radius:999px; height:6px;">
                                    <div style="background:{color}; width:{bar}%; height:6px; border-radius:999px;"></div>
                                </div>
                            </div>

                            <a href="{link}" target="_blank" style="background:#7c3aed; color:white; font-size:12px; font-weight:600; padding:8px 16px; border-radius:8px; text-decoration:none; display:inline-block;">▶ Watch on YouTube</a>
                        </div>

                        <!-- Thumbnail -->
                        <div style="width:260px; flex-shrink:0; background:#181825; display:flex; align-items:center; justify-content:center; padding:16px;">
                            <a href="{link}" target="_blank">
                                <img src="{thumb}" style="width:100%; border-radius:10px; object-fit:cover;" />
                            </a>
                        </div>
                    </div>

                    <!-- Iframe Preview -->
                    <div style="padding:0 20px 20px 20px;">
                        <div style="font-size:11px; color:#585b70; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.5px;">Preview</div>
                        <iframe
                            width="100%"
                            height="315"
                            src="{embed_url}"
                            frameborder="0"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowfullscreen
                            style="border-radius:10px; display:block;">
                        </iframe>
                    </div>
                </div>
                </body>
                </html>
                """, height=650)