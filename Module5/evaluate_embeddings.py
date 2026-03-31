import pandas as pd
import numpy as np
import warnings

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

print("\nSTEP 1 — Loading datasets\n")

videos = pd.read_csv("../Module3/video_with_transcripts.csv")

queries = pd.read_csv("../Module4/query_video_mapping.csv")
# Fix column naming issue
queries = queries.rename(columns={
    "relevant_video_id":"video_id"
})


print("Videos:",videos.shape)
print("Queries:",queries.shape)


required_cols=["video_id","title","transcript"]

for col in required_cols:
    if col not in videos.columns:
        raise Exception(f"Missing column {col}")

if "video_id" not in queries.columns:
    raise Exception("Query mapping must contain video_id")

print("Dataset check complete")


print("\nSTEP 2 — Loading embedding models\n")

models={

"MiniLM":"all-MiniLM-L6-v2",

"MPNet":"all-mpnet-base-v2",

"MultiQA":"multi-qa-MiniLM-L6-cos-v1"

}

metrics=[

"cosine",

"dot",

"euclidean",

"cityblock",

"chebyshev"

]

evaluation_results=[]
top_k_results=[]
query_eval=[]

for model_name,model_path in models.items():
    print("\nLoading model:",model_name)
    model=SentenceTransformer(model_path)
    print("STEP 3 — Creating video embeddings")
    
    video_text=(
        videos["title"].fillna("")+
        " "+
        videos["transcript"].fillna("")
    ).tolist()
    
    video_embeddings=model.encode(
        video_text,
        show_progress_bar=True
    )

    print("STEP 4 — Creating query embeddings")

    query_list=queries["query"].tolist()
    query_embeddings=model.encode(query_list)
    for metric in metrics:
        print("Evaluating Metric:",metric)
        if metric=="cosine":
            scores=cosine_similarity(
                query_embeddings,
                video_embeddings
            )

            ranked=np.argsort(-scores,axis=1)

        elif metric=="dot":
            scores=np.dot(
                query_embeddings,
                video_embeddings.T
            )
            ranked=np.argsort(-scores,axis=1)

        else:
            distances=cdist(
                query_embeddings,
                video_embeddings,
                metric=metric
            )

            ranked=np.argsort(distances,axis=1)

        print("STEP 7 — Ranking videos")

        top1=0
        top3=0
        top5=0

        ranks=[]

        for i,query in enumerate(query_list):

            expected=queries.iloc[i]["video_id"]
            ranking=ranked[i]
            top5_ids=ranking[:5]

            for r,idx in enumerate(top5_ids):

                if metric in ["cosine","dot"]:

                    score=scores[i][idx]

                else:

                    score=distances[i][idx]

                top_k_results.append([
                model_name,
                metric,
                query,
                r+1,
                videos.iloc[idx]["video_id"],
                score
                ])

            retrieved_rank=len(videos)

            for pos,idx in enumerate(ranking):

                if videos.iloc[idx]["video_id"]==expected:

                    retrieved_rank=pos+1
                    break
                
            ranks.append(retrieved_rank)

            if retrieved_rank==1:
                top1+=1

            if retrieved_rank<=3:
                top3+=1

            if retrieved_rank<=5:
                top5+=1

            query_eval.append([
            model_name,
            metric,
            query,
            expected,
            retrieved_rank
            ])

        total=len(query_list)

        evaluation_results.append([
        model_name,
        metric,
        top1/total,
        top3/total,
        top5/total,
        np.mean(ranks)
        ])

print("\nSTEP 8 — Creating evaluation tables")

eval_df=pd.DataFrame(
evaluation_results,
columns=[
"Model",
"Metric",
"Top-1 Recall",
"Top-3 Recall",
"Top-5 Recall",
"Avg Rank"
]
)

topk_df=pd.DataFrame(
top_k_results,
columns=[
"Model",
"Metric",
"Query",
"Rank",
"Video",
"Score"
]
)

query_df=pd.DataFrame(
query_eval,
columns=[
"Model",
"Metric",
"Query",
"Expected Video",
"Retrieved Rank"
]
)

print("\nFINAL RESULTS:\n")
final=eval_df.sort_values(
["Top-3 Recall","Top-1 Recall"],
ascending=False
)

print(final.to_string(index=False))

print("\nFiles saved:")
eval_df.to_csv(
"../Module5/evaluation_results.csv",
index=False
)
topk_df.to_csv(
"../Module5/top_k_results.csv",
index=False
)
query_df.to_csv(
"../Module5/query_evaluation.csv",
index=False
)

print("\nBEST CONFIGURATION:")
best=final.iloc[0]
print("Model:",best["Model"])
print("Metric:",best["Metric"])
print("Top-3 Recall:",round(best["Top-3 Recall"],3))
print("Avg Rank:",round(best["Avg Rank"],2))
print("\nReason:")
print("Highest Top-3 recall with best ranking performance.")
