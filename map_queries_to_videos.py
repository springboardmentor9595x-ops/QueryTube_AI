import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading datasets...")

queries_df = pd.read_csv("search_queries.csv")
videos_df = pd.read_csv("cleaned_transcripts.csv")

print("Queries:", len(queries_df))
print("Videos:", len(videos_df))

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding video transcripts...")

video_embeddings = model.encode(
    videos_df["transcript"].tolist(),
    show_progress_bar=True
)

print("Encoding queries...")

query_embeddings = model.encode(
    queries_df["query"].tolist(),
    show_progress_bar=True
)

print("Calculating similarity...")

similarity = cosine_similarity(query_embeddings, video_embeddings)

mapped_video_ids = []

for row in similarity:
    best_index = row.argmax()
    video_id = videos_df.iloc[best_index]["video_id"]
    mapped_video_ids.append(video_id)

queries_df["relevant_video_id"] = mapped_video_ids

queries_df.to_csv("query_video_mapping.csv", index=False)

print("Query mapping completed.")
print("Saved file: query_video_mapping.csv")