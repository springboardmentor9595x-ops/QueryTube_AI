# Step 0: Imports
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# Load data
df = pd.read_csv("../data/cleaned_transcripts.csv")
queries_df = pd.read_csv("../data/query_video_mapping.csv")

# Model
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

# Embeddings
titles = df['title'].astype(str).tolist()
transcripts = df['transcript'].astype(str).tolist()
queries = queries_df['query'].astype(str).tolist()

title_embeddings = model.encode(titles)
transcript_embeddings = model.encode(transcripts)
query_embeddings = model.encode(queries)

# Similarity
cosine_sim = cosine_similarity(query_embeddings, transcript_embeddings)

# Distance
# -----------------------------
# Step 6: Distance-Based Ranking
# -----------------------------

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

euclidean_dist = euclidean_distances(query_embeddings, transcript_embeddings)
manhattan_dist = manhattan_distances(query_embeddings, transcript_embeddings)

chebyshev_dist = np.max(
    np.abs(query_embeddings[:, np.newaxis] - transcript_embeddings),
    axis=2
)

# -----------------------------
# Euclidean Table
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
euclidean_df.to_csv("../data/multiqa_euclidean_results.csv", index=False)

print("Euclidean CSV saved ✅")


# -----------------------------
# Manhattan Table
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
manhattan_df.to_csv("../data/multiqa_manhattan_results.csv", index=False)

print("Manhattan CSV saved ✅")


# -----------------------------
# Chebyshev Table
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
chebyshev_df.to_csv("../data/multiqa_chebyshev_results.csv", index=False)

print("Chebyshev CSV saved ✅")

# Ranking
results = []
for i, q in enumerate(queries):
    idxs = np.argsort(cosine_sim[i])[::-1][:5]
    for r, idx in enumerate(idxs, 1):
        results.append({
            "Query": q,
            "Rank": r,
            "Video": df.iloc[idx]['video_id'],
            "Score": cosine_sim[i][idx]
        })

final_df = pd.DataFrame(results)
final_df.to_csv("../data/multiqa_cosine_results.csv", index=False)

# Evaluation
evals = []
for q in queries:
    true = queries_df[queries_df['query'] == q]['relevant_video_id'].values[0]
    retrieved = final_df[final_df['Query'] == q]

    match = retrieved[retrieved['Video'] == true]
    rank = int(match['Rank'].values[0]) if not match.empty else None

    evals.append({"Query": q, "Retrieved Rank": rank})

eval_df = pd.DataFrame(evals)

# Metrics
total = len(eval_df)

metrics_df = pd.DataFrame([{
    "Model": "MultiQA",
    "Top1": sum(eval_df['Retrieved Rank'] == 1)/total,
    "Top3": sum(eval_df['Retrieved Rank'].apply(lambda x: x and x <= 3))/total,
    "Top5": sum(eval_df['Retrieved Rank'].apply(lambda x: x and x <= 5))/total,
    "AvgRank": eval_df['Retrieved Rank'].dropna().mean()
}])

metrics_df.to_csv("../data/multiqa_metrics.csv", index=False)