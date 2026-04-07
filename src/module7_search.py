import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/video_index.csv")

model = SentenceTransformer("all-mpnet-base-v2")

embeddings = np.array(df["embedding"].apply(eval).tolist())

def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower())

def search(query, top_k=5):

    clean_query = clean_text(query)

    query_embedding = model.encode([clean_query])
    scores = cosine_similarity(query_embedding, embeddings)[0]

    stopwords = ["what", "is", "how", "the", "a", "an", "in", "of", "to"]
    query_words = [w for w in clean_query.split() if w not in stopwords]

    results = []

    for idx in range(len(df)):
        
        title = clean_text(df.iloc[idx]["title"])
        score = scores[idx]

        if len(query_words) > 0:
            keyword_match_count = sum(1 for word in query_words if word in title)
            keyword_score = keyword_match_count / len(query_words)
        else:
            keyword_score = 0

        intent_boost = 0

        if clean_query.startswith("what is"):
            if "what is" in title or "introduction" in title or "explained" in title:
                intent_boost += 0.1

        if any(word in clean_query for word in ["roadmap", "learn", "course", "tutorial", "developer"]):
            if any(word in title for word in ["course", "tutorial", "full", "learn", "developer", "career"]):
                intent_boost += 0.2

        if any(word in title for word in ["tool", "vite", "webpack"]):
            intent_boost -= 0.1

        if len(query_words) <= 2:
            final_score = (0.4 * score) + (0.5 * keyword_score) + (0.1 * intent_boost)
        else:
            final_score = (0.6 * score) + (0.25 * keyword_score) + (0.15 * intent_boost)

        results.append({
            "video_id": df.iloc[idx]["video_id"],
            "title": df.iloc[idx]["title"],
            "score": round(float(final_score), 3)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return pd.DataFrame(results[:top_k])

query = input("Enter your search query: ")

results = search(query)

print("\nQuery:", query)
print("Total Videos:", len(df))

print("\nTop Result:")
print(results.head(1))

print("\nOther Results:")
print(results.iloc[1:])