import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

with open("best_config.json", "r") as f:
    _cfg = json.load(f)

MODEL_NAME    = _cfg.get("model_name",         "all-MiniLM-L6-v2")
DEFAULT_METRIC = _cfg.get("metric",            "cosine")
DEFAULT_TW     = _cfg.get("title_weight",       0.3)
DEFAULT_TRW    = _cfg.get("transcript_weight",  0.7)
DEFAULT_TOPK   = _cfg.get("top_k",              5)

print("[search_engine] Loading video index...")
_index_df      = pd.read_csv("video_index.csv")
_emb_cols      = [c for c in _index_df.columns if c.startswith("embedding_")]
doc_embeddings = _index_df[_emb_cols].values          
video_ids      = _index_df["video_id"].tolist()
titles         = _index_df["title"].tolist()

print(f"[search_engine] Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print("[search_engine] Encoding title embeddings...")
_clean_df = pd.read_csv("cleaned_transcripts.csv")
_clean_df["title"]      = _clean_df["title"].fillna("")
_clean_df["transcript"] = _clean_df["transcript"].fillna("")

title_embs = model.encode(
    _clean_df["title"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

print("[search_engine] Encoding transcript embeddings...")
trans_embs = model.encode(
    _clean_df["transcript"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

print("[search_engine] ✅ Ready.\n")

def _compute_scores(query_emb: np.ndarray,
                    metric: str,
                    title_weight: float,
                    transcript_weight: float) -> np.ndarray:
    if metric == "cosine":
        t  = cosine_similarity(query_emb, title_embs)[0]
        tr = cosine_similarity(query_emb, trans_embs)[0]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return title_weight * t + transcript_weight * tr


def returnSearchResults(query: str,
                        metric: str       = DEFAULT_METRIC,
                        title_weight: float = DEFAULT_TW,
                        top_k: int        = DEFAULT_TOPK,
                        threshold: float  = 0.15) -> list[dict]:
    if not query.strip():
        return []

    transcript_weight = 1.0 - title_weight
    query_emb = model.encode([query], convert_to_numpy=True)
    scores    = _compute_scores(query_emb, metric, title_weight, transcript_weight)

    ranked_idx = np.argsort(scores)[::-1]
    results = []
    for idx in ranked_idx:
        score = float(scores[idx])
        if score < threshold:
            continue
        results.append({
            "video_id"    : video_ids[idx],
            "title"       : titles[idx],
            "score"       : round(score, 4),
            "youtube_link": f"https://www.youtube.com/watch?v={video_ids[idx]}",
        })
        if len(results) == int(top_k):
            break

    return results