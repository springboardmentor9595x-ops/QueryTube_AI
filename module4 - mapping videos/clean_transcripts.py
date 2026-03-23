import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

# ================================
# File Paths (FINAL INPUTS)
# ================================

VIDEO_FILE = "../module4/cleaned_transcripts.csv"
QUERY_FILE = "../module4/search_queries.csv"
MAPPING_FILE = "../module4/query_video_mapping.csv"

# ================================
# Step 1: Load Data
# ================================

print("Loading datasets...\n")

videos = pd.read_csv(VIDEO_FILE)
queries = pd.read_csv(QUERY_FILE)
mapping = pd.read_csv(MAPPING_FILE)

# Merge query + ground truth
queries = queries.merge(mapping, on="query")

print("Datasets loaded successfully!")
print(f"Total Videos: {len(videos)}")
print(f"Total Queries: {len(queries)}\n")

# ================================
# Step 2: Load Models
# ================================

models = {
    "MiniLM": SentenceTransformer("all-MiniLM-L6-v2"),
    "MPNet": SentenceTransformer("all-mpnet-base-v2"),
    "MultiQA": SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
}

# ================================
# Helper Functions
# ================================

def compute_similarity(query_emb, doc_embs, metric):
    if metric == "cosine":
        return cosine_similarity([query_emb], doc_embs)[0]
    elif metric == "dot":
        return np.dot(doc_embs, query_emb)

def compute_distance(query_emb, doc_embs, metric):
    return cdist([query_emb], doc_embs, metric=metric)[0]

def get_rank(ranked_ids, true_id):
    try:
        return ranked_ids.index(true_id) + 1
    except ValueError:
        return None

# ================================
# Step 3–9: Evaluation
# ================================

results = []

# Combine title + transcript
videos["text"] = videos["title"].fillna('') + " " + videos["transcript"].fillna('')

for model_name, model in models.items():
    print(f"\nRunning Model: {model_name}")

    # Encode videos
    video_embeddings = model.encode(
        videos["text"].tolist(),
        show_progress_bar=True
    )

    # Encode queries
    query_embeddings = model.encode(queries["query"].tolist())

    # Metrics
    similarity_metrics = ["cosine", "dot"]
    distance_metrics = ["euclidean", "cityblock", "chebyshev"]

    for metric in similarity_metrics + distance_metrics:
        print(f"Evaluating Metric: {metric}")

        top1 = top3 = top5 = 0
        ranks = []

        for i, query_emb in enumerate(query_embeddings):
            true_vid = queries.iloc[i]["relevant_video_id"]

            # Compute scores
            if metric in similarity_metrics:
                scores = compute_similarity(query_emb, video_embeddings, metric)
                ranked_idx = np.argsort(scores)[::-1]
            else:
                scores = compute_distance(query_emb, video_embeddings, metric)
                ranked_idx = np.argsort(scores)

            ranked_ids = videos.iloc[ranked_idx]["video_id"].tolist()

            rank = get_rank(ranked_ids, true_vid)

            if rank is not None:
                ranks.append(rank)

                if rank == 1:
                    top1 += 1
                if rank <= 3:
                    top3 += 1
                if rank <= 5:
                    top5 += 1

        total = len(queries)

        results.append({
            "Model": model_name,
            "Metric": metric,
            "Top-1 Recall": round(top1 / total, 3),
            "Top-3 Recall": round(top3 / total, 3),
            "Top-5 Recall": round(top5 / total, 3),
            "Avg Rank": round(np.mean(ranks), 2) if ranks else None
        })

# ================================
# Step 9: Results Table
# ================================

results_df = pd.DataFrame(results)

print("\n📊 FINAL RESULTS:\n")
print(results_df.sort_values(by="Top-3 Recall", ascending=False))

# Save results
results_df.to_csv("evaluation_results.csv", index=False)

# ================================
# Step 10: Best Model Selection
# ================================

best = results_df.sort_values(by="Top-3 Recall", ascending=False).iloc[0]

print("\n🏆 BEST CONFIGURATION:")
print(f"Model: {best['Model']}")
print(f"Metric: {best['Metric']}")
print(f"Top-3 Recall: {best['Top-3 Recall']}")
print(f"Avg Rank: {best['Avg Rank']}")