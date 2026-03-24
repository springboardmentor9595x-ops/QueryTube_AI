import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

# ================================
# Step 1: Load datasets
# ================================

VIDEO_FILE = "../module4/cleaned_transcripts.csv"
QUERY_FILE = "../module4/search_queries.csv"
MAPPING_FILE = "../module4/query_video_mapping.csv"

print("Loading datasets...\n")

videos = pd.read_csv(VIDEO_FILE)
queries = pd.read_csv(QUERY_FILE)
mapping = pd.read_csv(MAPPING_FILE)

# Merge queries with ground truth
queries = queries.merge(mapping, on="query")

print(f"Videos: {len(videos)}, Queries: {len(queries)}")

# ================================
# Step 2: Load models
# ================================

models = {
    "MiniLM": SentenceTransformer("all-MiniLM-L6-v2"),
    "MPNet": SentenceTransformer("all-mpnet-base-v2"),
    "MultiQA": SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
}

# ================================
# Helper functions
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
# Prepare data
# ================================

videos["combined"] = videos["title"].fillna('') + " " + videos["transcript"].fillna('')

# ================================
# Step 3–9: Evaluation
# ================================

results_summary = []
top_k_results = []
query_eval_results = []

TOP_K = 5

for model_name, model in models.items():
    print(f"\nRunning Model: {model_name}")

    # Encode once per model
    video_embeddings = model.encode(videos["combined"].tolist(), show_progress_bar=True)
    query_embeddings = model.encode(queries["query"].tolist())

    similarity_metrics = ["cosine", "dot"]
    distance_metrics = ["euclidean", "cityblock", "chebyshev"]

    for metric in similarity_metrics + distance_metrics:
        print(f"Evaluating Metric: {metric}")

        top1 = top3 = top5 = 0
        ranks = []

        for i, query_emb in enumerate(query_embeddings):
            query_text = queries.iloc[i]["query"]
            true_vid = queries.iloc[i]["relevant_video_id"]

            # Compute scores
            if metric in similarity_metrics:
                scores = compute_similarity(query_emb, video_embeddings, metric)
                ranked_idx = np.argsort(scores)[::-1]
            else:
                scores = compute_distance(query_emb, video_embeddings, metric)
                ranked_idx = np.argsort(scores)

            ranked_ids = videos.iloc[ranked_idx]["video_id"].tolist()
            ranked_scores = scores[ranked_idx]

            # ================================
            # Step 7: Store Top-K results
            # ================================
            for rank_pos in range(TOP_K):
                top_k_results.append({
                    "Model": model_name,
                    "Metric": metric,
                    "Query": query_text,
                    "Rank": rank_pos + 1,
                    "Video": ranked_ids[rank_pos],
                    "Score": round(float(ranked_scores[rank_pos]), 4)
                })

            # ================================
            # Step 8: Query evaluation
            # ================================
            rank = get_rank(ranked_ids, true_vid)

            query_eval_results.append({
                "Model": model_name,
                "Metric": metric,
                "Query": query_text,
                "Expected Video": true_vid,
                "Retrieved Rank": rank
            })

            # Metrics calculation
            if rank is not None:
                ranks.append(rank)

                if rank == 1:
                    top1 += 1
                if rank <= 3:
                    top3 += 1
                if rank <= 5:
                    top5 += 1

        total = len(queries)

        results_summary.append({
            "Model": model_name,
            "Metric": metric,
            "Top-1 Recall": round(top1 / total, 3),
            "Top-3 Recall": round(top3 / total, 3),
            "Top-5 Recall": round(top5 / total, 3),
            "Avg Rank": round(np.mean(ranks), 2) if ranks else None
        })

# ================================
# Step 9: Save results
# ================================

results_df = pd.DataFrame(results_summary)
top_k_df = pd.DataFrame(top_k_results)
query_eval_df = pd.DataFrame(query_eval_results)

print("\n📊 FINAL RESULTS:\n")
print(results_df.sort_values(by="Top-3 Recall", ascending=False))

# Save all outputs
results_df.to_csv("evaluation_results.csv", index=False)
top_k_df.to_csv("top_k_results.csv", index=False)
query_eval_df.to_csv("query_evaluation.csv", index=False)

print("\n✅ Files saved:")
print("evaluation_results.csv")
print("top_k_results.csv")
print("query_evaluation.csv")

# ================================
# Step 10: Best model
# ================================

best = results_df.sort_values(by="Top-3 Recall", ascending=False).iloc[0]

print("\n🏆 BEST CONFIGURATION:")
print(f"Model: {best['Model']}")
print(f"Metric: {best['Metric']}")
print(f"Top-3 Recall: {best['Top-3 Recall']}")
print(f"Avg Rank: {best['Avg Rank']}")