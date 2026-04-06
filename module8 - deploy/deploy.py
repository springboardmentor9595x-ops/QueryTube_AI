# ==============================
# Module 8: Final Deployment
# ==============================

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# ==============================
# Config
# ==============================
MODEL_NAME = "all-MiniLM-L6-v2"
THRESHOLD = 0.4   
TOP_K = 5

# ==============================
# Load Model & Data
# ==============================
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading embeddings...")
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = np.array(data["embeddings"])
metadata = data["metadata"]

print("System ready ✅")

# ==============================
# Search Function
# ==============================
def search_videos(query):
    if not query.strip():
        return []

    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]

    results = sorted(
        [(i, score) for i, score in enumerate(scores)],
        key=lambda x: x[1],
        reverse=True
    )

    filtered = [r for r in results if r[1] >= THRESHOLD]
    final = filtered[:TOP_K] if len(filtered) >= TOP_K else results[:TOP_K]

    output = []
    for idx, score in final:
        video = metadata[idx]

        output.append({
            "title": video.get("title", ""),
            "video_id": video.get("video_id", ""),
            "description": video.get("description", ""),
            "score": round(float(score), 3),
            "link": f"https://www.youtube.com/watch?v={video.get('video_id', '')}"
        })

    return output


# ==============================
# Interface Function
# ==============================
def search_interface(query):
    results = search_videos(query)

    if not query.strip():
        return [""] * 20

    updates = []

    for i in range(5):
        if i < len(results):
            r = results[i]

            updates.extend([
                f"### 🎥 {r['title']}",
                f"[▶️ Watch Video]({r['link']})",
                f"⭐ **Score:** {r['score']}",
                f"📝 {r['description']}"
            ])
        else:
            updates.extend(["", "", "", ""])

    return updates

# ==============================
# Clear Function
# ==============================
def clear_all():
    return [""] + [""] * 20


# ==============================
# UI (NO EMPTY BOXES ISSUE)
# ==============================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    with gr.Row():

        # LEFT PANEL
        with gr.Column(scale=1):
            gr.Markdown("""
# 🎬 QueryTube AI

### 🚀 What this project does:
- Semantic search over YouTube videos  
- Uses AI embeddings  
- Matches meaning, not keywords  

### 🧠 How it works:
1. Query → vector  
2. Compare with videos  
3. Rank using similarity  

### 💡 Try:
- spongebob  
- jennifer  
- devil wears prada
""")

        # RIGHT PANEL
        with gr.Column(scale=2):

            query_input = gr.Textbox(
                label="🔍 Search",
                placeholder="Try: spongebob..."
            )

            with gr.Row():
                search_btn = gr.Button("🚀 Search", variant="primary")
                clear_btn = gr.Button("🧹 Clear")

            # ===== RESULTS =====
            components = []

            for i in range(5):
                with gr.Column(visible=True):
                    title = gr.Markdown()
                    link = gr.Markdown()
                    score = gr.Markdown()
                    desc = gr.Markdown()

                    components.extend([title, link, score, desc])

            search_btn.click(
                fn=search_interface,
                inputs=query_input,
                outputs=components
            )

            clear_btn.click(
                fn=clear_all,
                inputs=[],
                outputs=[query_input] + components
            )

# ==============================
# Launch
# ==============================
demo.launch()