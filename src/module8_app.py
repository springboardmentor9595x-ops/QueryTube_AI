import pandas as pd
import numpy as np
import re
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("data/video_index.csv")

# Load model
model = SentenceTransformer("all-mpnet-base-v2")

# Load embeddings
embeddings = np.array(df["embedding"].apply(eval).tolist())


def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())


def search(query, top_k=10):

    clean_query = clean_text(query)

    query_embedding = model.encode([clean_query])
    scores = cosine_similarity(query_embedding, embeddings)[0]

    stopwords = ["what", "is", "how", "the", "a", "an", "in", "of", "to"]
    query_words = [w for w in clean_query.split() if w not in stopwords]

    results = []

    for idx in range(len(df)):

        title_raw = df.iloc[idx]["title"]
        title = clean_text(title_raw)
        score = scores[idx]

        if len(query_words) > 0:
            keyword_match_count = sum(1 for word in query_words if word in title)
            keyword_score = keyword_match_count / len(query_words)
        else:
            keyword_score = 0

        intent_boost = 0

        if clean_query.startswith("what is"):
            if "what is" in title or "introduction" in title or "explained" in title:
                intent_boost += 0.1

        if any(word in clean_query for word in ["roadmap", "learn", "course", "tutorial", "developer"]):
            if any(word in title for word in ["course", "tutorial", "full", "learn", "developer", "career"]):
                intent_boost += 0.2

        if any(word in title for word in ["tool", "vite", "webpack"]):
            intent_boost -= 0.1

        if len(query_words) <= 2:
            final_score = (0.4 * score) + (0.5 * keyword_score) + (0.1 * intent_boost)
        else:
            final_score = (0.6 * score) + (0.25 * keyword_score) + (0.15 * intent_boost)

        video_id = df.iloc[idx]["video_id"]

        results.append({
            "title": title_raw,
            "score": round(float(final_score), 3),
            "video_id": video_id
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def search_ui(query):

    results = search(query)

    output = """
    <div style="
        display:grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap:20px;
        align-items:stretch;
    ">
    """

    for res in results:

        video_id = res["video_id"]
        thumbnail = f"https://img.youtube.com/vi/{video_id}/0.jpg"
        link = f"https://www.youtube.com/watch?v={video_id}"

        output += f"""
        <div style="
            background:#1e1e1e;
            border-radius:12px;
            overflow:hidden;
            box-shadow:0 4px 12px rgba(0,0,0,0.5);
            display:flex;
            flex-direction:column;
            height:100%;
            transition:transform 0.2s;
        "
        onmouseover="this.style.transform='scale(1.03)'"
        onmouseout="this.style.transform='scale(1)'"
        >

            <a href="{link}" target="_blank">
                <img src="{thumbnail}" 
                     style="width:100%; height:170px; object-fit:cover;">
            </a>

            <div style="
                padding:12px;
                flex-grow:1;
                display:flex;
                flex-direction:column;
                justify-content:space-between;
            ">

                <h4 style="
                    margin:0 0 10px 0;
                    font-size:15px;
                    line-height:1.3;
                    height:42px;
                    overflow:hidden;
                ">
                    {res['title']}
                </h4>

                <div>
                    <p style="margin:0; font-size:13px; color:#aaa;">
                        Relevance Score: {res['score']}
                    </p>

                    <a href="{link}" target="_blank"
                       style="color:#4da6ff; font-size:13px;">
                        Watch on YouTube
                    </a>
                </div>

            </div>

        </div>
        """

    output += "</div>"

    return output


# ✅ Gradio UI
with gr.Blocks() as app:

    gr.HTML("""
    <div style="display:flex; align-items:center; gap:10px;">
        
        <img src="https://cdn-icons-png.flaticon.com/512/3670/3670147.png"
             width="40">

        <h1 style="
            margin:0;
            font-size:28px;
            background: linear-gradient(to right, #4da6ff, #00ffcc);
            -webkit-background-clip: text;
            color: transparent;
        ">
            QueryTubeAI
        </h1>

    </div>

    <p style="margin-top:5px; color:gray;">
        Search videos using AI-powered semantic search
    </p>
    """)

    query_input = gr.Textbox(label="Enter your query")

    output = gr.HTML()

    query_input.submit(search_ui, inputs=query_input, outputs=output)

app.launch()