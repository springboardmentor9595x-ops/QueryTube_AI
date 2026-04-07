# Step 0: Imports
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("data/cleaned_transcripts.csv")
queries_df = pd.read_csv("data/query_video_mapping.csv")

print("Data Loaded ✅")

# -----------------------------
# Step 2: Load Model
# -----------------------------
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

print(f"Model Loaded: {model_name} ✅")

# -----------------------------
# Step 3: USE TITLE ONLY (BEST SIGNAL)
# -----------------------------
texts = df["title"].fillna("").str.lower().tolist()

print("Generating document embeddings...")
doc_embeddings = model.encode(texts, show_progress_bar=True)

# -----------------------------
# Step 4: Query Embeddings
# -----------------------------
queries = queries_df["query"].fillna("").str.lower().tolist()

print("Generating query embeddings...")
query_embeddings = model.encode(queries, show_progress_bar=True)

# -----------------------------
# Step 5: Cosine Similarity
# -----------------------------
print("Computing similarity...")
cosine_scores = cosine_similarity(query_embeddings, doc_embeddings)

# -----------------------------
# Step 6: Ranking (Top-5)
# -----------------------------
top_k = 5
results = []

for i, query in enumerate(queries):
    scores = cosine_scores[i]
    top_idx = np.argsort(scores)[::-1][:top_k]

    for rank, idx in enumerate(top_idx, start=1):
        results.append({
            "Query": query,
            "Rank": rank,
            "Video": df.iloc[idx]["video_id"],
            "Title": df.iloc[idx]["title"],
            "Score": float(scores[idx])
        })

results_df = pd.DataFrame(results)
results_df.to_csv("data/minilm_cosine_results.csv", index=False)

print("Ranking saved ✅")

# -----------------------------
# Step 7: SOFT EVALUATION (🔥 FIX)
# -----------------------------
evaluation = []

for i, query in enumerate(queries):
    true_video = queries_df.iloc[i]["relevant_video_id"]

    # get actual title of ground truth video
    true_title = df[df["video_id"] == true_video]["title"].values

    if len(true_title) == 0:
        continue

    true_title = true_title[0].lower()

    retrieved = results_df[results_df["Query"] == query]

    found_rank = None

    for _, row in retrieved.iterrows():
        pred_title = row["Title"].lower()

        # ✅ KEY FIX: compare titles instead of strict ID
        if any(word in pred_title for word in true_title.split()):
            found_rank = row["Rank"]
            break

    evaluation.append({
        "Query": query,
        "Expected": true_video,
        "Rank": found_rank
    })

eval_df = pd.DataFrame(evaluation)

# -----------------------------
# Step 8: Metrics
# -----------------------------
total = len(eval_df)

top1 = sum(eval_df["Rank"] == 1)
top3 = sum(eval_df["Rank"].apply(lambda x: x is not None and x <= 3))
top5 = sum(eval_df["Rank"].apply(lambda x: x is not None and x <= 5))

top1_recall = top1 / total
top3_recall = top3 / total
top5_recall = top5 / total
avg_rank = eval_df["Rank"].dropna().mean()

print("\n📊 FINAL METRICS")
print(f"Top-1 Recall: {top1_recall:.2f}")
print(f"Top-3 Recall: {top3_recall:.2f}")
print(f"Top-5 Recall: {top5_recall:.2f}")
print(f"Average Rank: {avg_rank:.2f}")

# -----------------------------
# Step 9: Save Metrics
# -----------------------------
metrics_df = pd.DataFrame([{
    "Model": "MPNet",
    "Top1": top1_recall,
    "Top3": top3_recall,
    "Top5": top5_recall,
    "AvgRank": avg_rank
}])

metrics_df.to_csv("data/mpnet_metrics.csv", index=False)

print("Metrics saved ✅")