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

model_name = "all-MiniLM-L6-v2"
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
print("\nTitle Embeddings Shape:", title_embeddings.shape)
print("Transcript Embeddings Shape:", transcript_embeddings.shape)

# -----------------------------
# Step 4: Generate Query Embeddings
# -----------------------------

queries = queries_df['query'].astype(str).tolist()

print("\nGenerating query embeddings... ⏳")

query_embeddings = model.encode(queries, show_progress_bar=True)

print("Query embeddings generated successfully ✅")
print("Query Embeddings Shape:", query_embeddings.shape)

# -----------------------------
# Step 5: Similarity Ranking
# -----------------------------

print("\nComputing similarity scores... ⏳")

cosine_sim_transcripts = cosine_similarity(query_embeddings, transcript_embeddings)
dot_sim_transcripts = np.dot(query_embeddings, transcript_embeddings.T)

print("Similarity metrics computed ✅")

# -----------------------------
# Step 6: Distance-Based Ranking
# -----------------------------

print("\nComputing distance metrics... ⏳")

euclidean_dist = euclidean_distances(query_embeddings, transcript_embeddings)
manhattan_dist = manhattan_distances(query_embeddings, transcript_embeddings)

chebyshev_dist = np.max(
    np.abs(query_embeddings[:, np.newaxis] - transcript_embeddings),
    axis=2
)

print("Distance metrics computed ✅")

# -----------------------------
# Step 7: Final Ranking Table (Cosine)
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

print("\nTop 5 Ranked Videos (Cosine) ✅")
print(final_df.head(15))

# ✅ SAVE COSINE RESULTS
final_df.to_csv("../data/minilm_cosine_results.csv", index=False)

# -----------------------------
# Distance Tables (Example: Euclidean)
# -----------------------------

euclidean_results = []

for i, query in enumerate(queries):
    scores = euclidean_dist[i]
    top_indices = np.argsort(scores)[:5]

    for rank, idx in enumerate(top_indices, start=1):
        euclidean_results.append({
            "Query": query,
            "Rank": rank,
            "Video": df.iloc[idx]['video_id'],
            "Distance": round(float(scores[idx]), 4)
        })

euclidean_df = pd.DataFrame(euclidean_results)

print("\nEuclidean Results ✅")
print(euclidean_df.head(10))

# ✅ SAVE EUCLIDEAN
euclidean_df.to_csv("../data/minilm_euclidean_results.csv", index=False)

# -----------------------------
# Manhattan Distance Table
# -----------------------------

manhattan_results = []

for i, query in enumerate(queries):
    scores = manhattan_dist[i]
    top_indices = np.argsort(scores)[:5]

    for rank, idx in enumerate(top_indices, start=1):
        manhattan_results.append({
            "Query": query,
            "Rank": rank,
            "Video": df.iloc[idx]['video_id'],
            "Distance": round(float(scores[idx]), 4)
        })

manhattan_df = pd.DataFrame(manhattan_results)

print("\nManhattan Results ✅")
print(manhattan_df.head(10))

# ✅ SAVE
manhattan_df.to_csv("../data/minilm_manhattan_results.csv", index=False)

# -----------------------------
# Chebyshev Distance Table
# -----------------------------

chebyshev_results = []

for i, query in enumerate(queries):
    scores = chebyshev_dist[i]
    top_indices = np.argsort(scores)[:5]

    for rank, idx in enumerate(top_indices, start=1):
        chebyshev_results.append({
            "Query": query,
            "Rank": rank,
            "Video": df.iloc[idx]['video_id'],
            "Distance": round(float(scores[idx]), 4)
        })

chebyshev_df = pd.DataFrame(chebyshev_results)

print("\nChebyshev Results ✅")
print(chebyshev_df.head(10))

# ✅ SAVE
chebyshev_df.to_csv("../data/minilm_chebyshev_results.csv", index=False)

# -----------------------------
# Step 8: Evaluation
# -----------------------------

print("\nEvaluating retrieval performance... ⏳")

evaluation_results = []

for query in queries:
    true_row = queries_df[queries_df['query'] == query]

    if len(true_row) == 0:
        continue

    # ✅ FIXED COLUMN NAME
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

print("\nEvaluation Table ✅")
print(eval_df.head())

# -----------------------------
# Metrics Calculation
# -----------------------------

total_queries = len(eval_df)

top1 = sum(eval_df['Retrieved Rank'] == 1)
top3 = sum(eval_df['Retrieved Rank'].apply(lambda x: x is not None and x <= 3))
top5 = sum(eval_df['Retrieved Rank'].apply(lambda x: x is not None and x <= 5))

top1_recall = top1 / total_queries
top3_recall = top3 / total_queries
top5_recall = top5 / total_queries

avg_rank = eval_df['Retrieved Rank'].dropna().mean()

print("\n📊 Evaluation Metrics")
print(f"Top-1 Recall: {top1_recall:.2f}")
print(f"Top-3 Recall: {top3_recall:.2f}")
print(f"Top-5 Recall: {top5_recall:.2f}")
print(f"Average Rank: {avg_rank:.2f}")

# ✅ SAVE METRICS (VERY IMPORTANT)
metrics_df = pd.DataFrame([{
    "Model": "MiniLM",
    "Top1": top1_recall,
    "Top3": top3_recall,
    "Top5": top5_recall,
    "AvgRank": avg_rank
}])

metrics_df.to_csv("../data/minilm_metrics.csv", index=False)
