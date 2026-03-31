import pandas as pd
import numpy as np
import warnings

from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# STEP 1 — Select best model 

MODEL_NAME = "all-MiniLM-L6-v2"

print("Selected Model:",MODEL_NAME)


# STEP 2 — Load cleaned dataset

videos = pd.read_csv("cleaned_transcripts.csv")

print("Dataset shape:",videos.shape)

required_cols=['video_id','title','datetime','transcript']

for col in required_cols:
    if col not in videos.columns:
        raise Exception(f"Missing column {col}")

print("Dataset validation complete")


# STEP 3 — Load SentenceTransformer model

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully")

# STEP 4 — Generating Title embeddings

titles = videos["title"].fillna("").tolist()

title_embeddings = model.encode(

    titles,

    show_progress_bar=True

)

print("Title embedding shape:",title_embeddings.shape)

# STEP 5 — Generating Transcript embeddings

transcripts = videos["transcript"].fillna("").tolist()

transcript_embeddings = model.encode(

    transcripts,

    show_progress_bar=True

)

print("Transcript embedding shape:",transcript_embeddings.shape)


# STEP 6 — Combine embeddings (average)

combined_embeddings = (

    title_embeddings + transcript_embeddings

) / 2

print("Combined embedding shape:",combined_embeddings.shape)


# STEP 7 — Attaching embeddings to dataset

embedding_df = pd.DataFrame(

combined_embeddings

)

embedding_df.columns=[

f"emb_{i}" for i in range(combined_embeddings.shape[1])

]


video_index = pd.concat(

[videos,embedding_df],

axis=1

)

print("Final dataset shape:",video_index.shape)


# STEP 8 — Save video index

video_index.to_csv(

"video_index.csv",

index=False

)

print("File saved: video_index.csv")


# STEP 9 — Validating embeddings

rows_ok = len(video_index)==len(videos)

nulls = embedding_df.isnull().sum().sum()

dim = combined_embeddings.shape[1]


print("Row check:",rows_ok)

print("Null embeddings:",nulls)

print("Embedding dimension:",dim)


if rows_ok and nulls==0:
    print("Embedding validation passed")
else:
    print("Validation issues detected")
