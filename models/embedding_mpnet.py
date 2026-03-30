# Step 0: Imports
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# -----------------------------
# Step 1: Load Cleaned Dataset
# -----------------------------

data_path = "../data/cleaned_transcripts.csv"
df = pd.read_csv(data_path)

print("Dataset Loaded Successfully ✅")
print(df.head())
print(df.columns)

queries_path = "../data/query_video_mapping.csv"
queries_df = pd.read_csv(queries_path)

print("\nQueries Loaded Successfully ✅")
print(queries_df.head())

# -----------------------------
# Step 2: Load Embedding Model
# -----------------------------

model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

print(f"\nModel Loaded: {model_name} ✅")

# -----------------------------
# Step 3: Generate Embeddings
# -----------------------------

titles = df['title'].astype(str).tolist()
transcripts = df['transcript'].astype(str).tolist()

print("\nGenerating embeddings... ⏳")

title_embeddings = model.encode(titles, show_progress_bar=True)
transcript_embeddings = model.encode(transcripts, show_progress_bar=True)

print("Embeddings generated successfully ✅")

# -----------------------------
# Step 4: Generate Query Embeddings
# -----------------------------

queries = queries_df['query'].astype(str).tolist()

print("\nGenerating query embeddings... ⏳")

query_embeddings = model.encode(queries, show_progress_bar=True)

print("Query embeddings generated successfully ✅")

# -----------------------------
# Step 5: Similarity Ranking
# -----------------------------

cosine_sim_transcripts = cosine_similarity(query_embeddings, transcript_embeddings)
dot_sim_transcripts = np.dot(query_embeddings, transcript_embeddings.T)

# -----------------------------
# Step 6: Distance-Based Ranking
# -----------------------------

euclidean_dist = euclidean_distances(query_embeddings, transcript_embeddings)
manhattan_dist = manhattan_distances(query_embeddings, transcript_embeddings)

chebyshev_dist = np.max(
    np.abs(query_embeddings[:, np.newaxis] - transcript_embeddings),
    axis=2
)

# -----------------------------
# Step 7: Ranking (Cosine)
# -----------------------------

top_k = 5
final_results = []

for i, query in enumerate(queries):
    scores = cosine_sim_transcripts[i]
    top_indices = np.argsort(scores)[::-1][:top_k]

    for rank, idx in enumerate(top_indices, start=1):
        final_results.append({
            "Query": query,
            "Rank": rank,
            "Video": df.iloc[idx]['video_id'],
            "Score": round(float(scores[idx]), 4)
        })

final_df = pd.DataFrame(final_results)
final_df.to_csv("../data/mpnet_cosine_results.csv", index=False)

# -----------------------------
# Distance Tables
# -----------------------------

def create_distance_df(dist_matrix, name):
    results = []
    for i, query in enumerate(queries):
        scores = dist_matrix[i]
        top_indices = np.argsort(scores)[:5]

        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "Query": query,
                "Rank": rank,
                "Video": df.iloc[idx]['video_id'],
                "Distance": round(float(scores[idx]), 4)
            })
    df_out = pd.DataFrame(results)
    df_out.to_csv(f"../data/mpnet_{name}_results.csv", index=False)

create_distance_df(euclidean_dist, "euclidean")
create_distance_df(manhattan_dist, "manhattan")
create_distance_df(chebyshev_dist, "chebyshev")

# -----------------------------
# Step 8: Evaluation
# -----------------------------

evaluation_results = []

for query in queries:
    true_row = queries_df[queries_df['query'] == query]
    if len(true_row) == 0:
        continue

    true_video = true_row['relevant_video_id'].values[0]
    retrieved = final_df[final_df['Query'] == query]

    match = retrieved[retrieved['Video'] == true_video]
    rank = int(match['Rank'].values[0]) if not match.empty else None

    evaluation_results.append({
        "Query": query,
        "Expected Video": true_video,
        "Retrieved Rank": rank
    })

eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv("../data/mpnet_evaluation.csv", index=False)

# -----------------------------
# Metrics
# -----------------------------

total = len(eval_df)

top1 = sum(eval_df['Retrieved Rank'] == 1)
top3 = sum(eval_df['Retrieved Rank'].apply(lambda x: x is not None and x <= 3))
top5 = sum(eval_df['Retrieved Rank'].apply(lambda x: x is not None and x <= 5))

metrics_df = pd.DataFrame([{
    "Model": "MPNet",
    "Top1": top1/total,
    "Top3": top3/total,
    "Top5": top5/total,
    "AvgRank": eval_df['Retrieved Rank'].dropna().mean()
}])

metrics_df.to_csv("../data/mpnet_metrics.csv", index=False)