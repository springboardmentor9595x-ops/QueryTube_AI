import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, cityblock


# =========================
# 🔹 Load CSV Data
# =========================
def load_data(csv_path, model):
    df = pd.read_csv(csv_path)

    print("Columns in CSV:", df.columns)

    # Fill missing values
    df['title'] = df['title'].fillna("")
    df['transcript'] = df['transcript'].fillna("")

    print("Generating embeddings... (this may take time)")

    # Generate embeddings (row-wise)
    df['title_embedding'] = df['title'].apply(lambda x: model.encode(x))
    df['transcript_embedding'] = df['transcript'].apply(lambda x: model.encode(x))

    print("Embeddings generated!")

    return df


# =========================
# 🔹 Load Model
# =========================
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# 🔹 Encode Query
# =========================
def encode_query(query, model):
    return model.encode(query)


# =========================
# 🔹 Similarity Functions
# =========================
def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0][0]


def euclidean_sim(a, b):
    return 1 / (1 + euclidean(a, b))


def manhattan_sim(a, b):
    return 1 / (1 + cityblock(a, b))


# =========================
# 🔹 Combined Score
# =========================
def combined_score(query_emb, title_emb, transcript_emb,
                   metric="cosine",
                   w_title=0.3,
                   w_transcript=0.7):

    if metric == "cosine":
        sim_func = cosine_sim
    elif metric == "euclidean":
        sim_func = euclidean_sim
    elif metric == "manhattan":
        sim_func = manhattan_sim
    else:
        raise ValueError("Invalid metric")

    title_score = sim_func(query_emb, title_emb)
    transcript_score = sim_func(query_emb, transcript_emb)

    return w_title * title_score + w_transcript * transcript_score


# =========================
# 🔹 Search Function
# =========================
def returnSearchResults(query, df, model,
                        top_k=5,
                        threshold=0.3,
                        metric="cosine"):

    query_emb = encode_query(query, model)

    scores = []

    for _, row in df.iterrows():
        score = combined_score(
            query_emb,
            row['title_embedding'],
            row['transcript_embedding'],
            metric
        )
        scores.append(score)

    df_copy = df.copy()
    df_copy['score'] = scores

    # Apply threshold
    df_filtered = df_copy[df_copy['score'] >= threshold]

    # Sort + Top-K
    results = df_filtered.sort_values(by='score', ascending=False).head(top_k)

    return results


# =========================
# 🔹 Evaluation Function (FIXED)
# =========================
def evaluate_search(query, df, model):

    top_k_values = [3, 5]
    thresholds = [0.4, 0.5, 0.6]  
    metrics = ["cosine", "euclidean", "manhattan"]

    print("\nEvaluating ALL metrics...\n")

    overall_best = None
    overall_best_score = -1

    for metric in metrics:

        print(f"🔍 Evaluating Metric: {metric.upper()}")
        print("#"*60)

        best_config = None
        best_score = -1

        for k in top_k_values:
            for t in thresholds:

                results = returnSearchResults(
                    query,
                    df,
                    model,
                    top_k=k,
                    threshold=t,
                    metric=metric
                )

                if results.empty:
                    continue

                avg_score = results['score'].mean()
                num_results = len(results)

                final_score = avg_score * num_results

                print("\n" + "-"*40)
                print(f"Metric: {metric}, Top-K: {k}, Threshold: {t}")
                print(f"Avg Score: {round(avg_score, 4)}, Results: {num_results}")

                if final_score > best_score:
                    best_score = final_score
                    best_config = (metric, k, t)

                if final_score > overall_best_score:
                    overall_best_score = final_score
                    overall_best = (metric, k, t)

        print("\n✅ Best for", metric.upper())

        if best_config is not None:
            print(f"Metric = {best_config[0]}")
            print(f"Top-K = {best_config[1]}")
            print(f"Threshold = {best_config[2]}")
        else:
            print("⚠️ No valid results found for this metric.")

    print("🏆 OVERALL BEST CONFIGURATION:")

    if overall_best is not None:
        print(f"Metric = {overall_best[0]}")
        print(f"Top-K = {overall_best[1]}")
        print(f"Threshold = {overall_best[2]}")
    else:
        print("⚠️ No valid configuration found.")

    print("="*60)


# =========================
# 🔹 MAIN
# =========================
if __name__ == "__main__":

    print("Loading data and model...")
    model = load_model()
    df = load_data("cleaned_transcripts.csv", model)

    print("Ready! Type your query (type 'exit' to quit)\n")

    while True:
        query = input("Enter your query: ")

        if query.lower() == "exit":
            print("Exiting...")
            break

        evaluate_search(query, df, model)