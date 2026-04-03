import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

# ================================
# Step 1: Load datasets
# ================================

VIDEO_FILE = "../module4 - mapping videos/cleaned_transcripts.csv"
QUERY_FILE = "../module4 - mapping videos/search_queries.csv"
MAPPING_FILE = "../module4 - mapping videos/query_video_mapping.csv"

print("Loading datasets...\n")

videos = pd.read_csv(VIDEO_FILE)
queries = pd.read_csv(QUERY_FILE)
mapping = pd.read_csv(MAPPING_FILE)

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

TOP_K = 5

for model_name, model in models.items():
    print(f"\nRunning Model: {model_name}")

    video_embeddings = model.encode(videos["combined"].tolist(), show_progress_bar=True)
    query_embeddings = model.encode(queries["query"].tolist())

    similarity_metrics = ["cosine", "dot"]
    distance_metrics = ["euclidean", "cityblock", "chebyshev"]

    for metric in similarity_metrics + distance_metrics:
        print(f"Evaluating Metric: {metric}")

        top1 = top3 = top5 = 0
        ranks = []

        for i, query_emb in enumerate(query_embeddings):
            true_vid = queries.iloc[i]["relevant_video_id"]

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

        results_summary.append({
            "Model": model_name,
            "Metric": metric,
            "Top-1 Recall": round(top1 / total, 3),
            "Top-3 Recall": round(top3 / total, 3),
            "Top-5 Recall": round(top5 / total, 3),
            "Avg Rank": round(np.mean(ranks), 2) if ranks else None
        })

# ================================
# Step 10: Save evaluation results
# ================================

results_df = pd.DataFrame(results_summary)

print("\n📊 FINAL RESULTS:\n")
print(results_df.sort_values(by="Top-3 Recall", ascending=False))

results_df.to_csv("evaluation_results.csv", index=False)

# ================================
# Step 11: Get BEST model
# ================================

best = results_df.sort_values(by="Top-3 Recall", ascending=False).iloc[0]

best_model_name = best["Model"]
best_metric = best["Metric"]

print("\n🏆 BEST CONFIGURATION:")
print(f"Model: {best_model_name}")
print(f"Metric: {best_metric}")

# ================================
# Step 12: Generate FINAL embeddings
# ================================

best_model = models[best_model_name]

print("\n🚀 Generating embeddings using BEST model...")

final_embeddings = best_model.encode(
    videos["combined"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

# ================================
# Step 13: Prepare metadata
# ================================

metadata = []

for _, row in videos.iterrows():
    metadata.append({
        "title": row.get("title", ""),
        "video_id": row.get("video_id", ""),
        "description": str(row.get("transcript", ""))[:200]
    })

# ================================
# Step 14: Save embeddings.pkl
# ================================

data = {
    "embeddings": final_embeddings,
    "metadata": metadata,
    "model": best_model_name,
    "metric": best_metric
}

with open("embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("\n✅ embeddings.pkl created successfully!")
print("📁 Ready for deployment 🚀")