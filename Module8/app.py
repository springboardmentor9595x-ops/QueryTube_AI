import pandas as pd
import numpy as np
import gradio as gr

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# MODEL + DATA
MODEL_NAME="all-mpnet-base-v2"

print("Loading model...")

model=SentenceTransformer(MODEL_NAME)

video_index=pd.read_csv("../Module6/video_index.csv")

embedding_cols=[c for c in video_index.columns if c.startswith("emb_")]

video_embeddings=video_index[embedding_cols].values

print(f"System ready ✓ ({len(video_index)} videos indexed)")


# SEARCH FUNCTION
def search(query,threshold,top_k):

    if not query.strip():
        return _empty("Enter a query to search")

    # Query embedding
    query_emb=model.encode([query])

    # Similarity calculation
    scores=cosine_similarity(
        query_emb,
        video_embeddings
    )[0]

    # Number of results
    top_k=int(top_k)

    # Always return best matches (no strict filtering)
    ranked=np.argsort(-scores)[:top_k]


    lines=[

        f"### Results for: *{query}*",

        f"> Found **{len(ranked)}** videos",

        ""

    ]


    for r,i in enumerate(ranked,1):

        row=video_index.iloc[i]

        title=row["title"]

        vid=row["video_id"]

        score=round(scores[i],3)

        percent=f"{score*100:.1f}%"


        bar="█"*int(score*10)+"░"*(10-int(score*10))


        link=f"https://youtube.com/watch?v={vid}"

        thumb=f"https://img.youtube.com/vi/{vid}/0.jpg"


        lines += [

        "---",

        f"#### {r}. {title}",

        f"Relevance Score `{bar}` {percent}",

        f"[![thumbnail]({thumb})]({link})",

        "",

        f"▶️ [Watch Video on YouTube]({link})",

        ""
        ]


    return "\n".join(lines)



def _empty(msg):

    return f"""

### No Results Found

{msg}

Try queries like:
• python tutorial  
• java arrays  
• numpy functions  

"""


# UI
theme=gr.themes.Base(
primary_hue=gr.themes.colors.green,
neutral_hue=gr.themes.colors.slate,
font=gr.themes.GoogleFont("IBM Plex Mono"),
font_mono=gr.themes.GoogleFont("IBM Plex Mono")
)


with gr.Blocks(theme=theme,title="YouTube Semantic Video Search") as demo:

    gr.Markdown("# 🎬 YouTube Semantic Video Search")


    with gr.Row():

        # LEFT PANEL
        with gr.Column(scale=4):

            gr.Markdown("## 🔍 Search Videos")

            query=gr.Textbox(
            placeholder="💡 Example: python decorators",
            info="Press Enter or click search"
            )


            with gr.Row():

                search_btn=gr.Button(
                "🔎 Search",
                variant="primary"
                )

                clear_btn=gr.Button("🧹 Clear")


            status=gr.Markdown("")


            gr.Markdown("## 📊 Results")

            results=gr.Markdown(
            value="Type something to start searching 🚀"
            )


        # RIGHT PANEL
        with gr.Column(scale=1):

            gr.Markdown("""
## 📘 About Project

Semantic video search for programming tutorials that finds relevant videos based on meaning rather than keywords.

### ⚙ How it works

🧠 MPNet converts text into embeddings  
📏 Cosine similarity measures relevance  
🏆 Videos ranked by similarity  
📺 Top results displayed
""")


            with gr.Accordion("⚙ Advanced Search Settings", open=False):

                threshold=gr.Slider(
                0.1,0.9,
                value=0.35,
                step=0.05,
                label="🎯 Similarity Strictness"
                )


                top_k=gr.Slider(
                1,10,
                value=5,
                step=1,
                label="📌 Number of Results"
                )

    inputs=[query,threshold,top_k]


    search_btn.click(

    fn=lambda:"Searching...",

    outputs=status

    ).then(

    fn=search,

    inputs=inputs,

    outputs=results

    ).then(

    fn=lambda:"",

    outputs=status

    )


    query.submit(
    fn=search,
    inputs=inputs,
    outputs=results
    )


    clear_btn.click(
    fn=lambda:("","> Enter a query and press Search"),
    outputs=[query,results]
    )


demo.launch()