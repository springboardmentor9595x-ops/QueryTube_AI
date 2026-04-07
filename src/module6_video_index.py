import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

df = pd.read_csv("data/cleaned_transcripts.csv")

print("Dataset Loaded ✅")
print("Shape:", df.shape)

print("\nLoading embedding model...")
# Model
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

print(f"Model Loaded: {model_name} ✅")

print("\nPreparing text for embedding...")

# Texts
texts = (df["title"] + " " + df["transcript"].str[:500]).fillna("").tolist()

print("Sample text:")
print(texts[0][:200])

print("\nGenerating embeddings... ⏳")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embeddings generated ✅")
print("Embedding shape:", embeddings.shape)

print("\nAttaching embeddings to dataset...")

df["embedding"] = embeddings.tolist()

print("Embeddings added to DataFrame ✅")

output_path = "data/video_index.csv"

df.to_csv(output_path, index=False)

print(f"\nVideo index saved at: {output_path} ✅")

print("\nRunning validation checks...")

assert len(df) == len(embeddings)

sample_embedding = df["embedding"].iloc[0]
print("Sample embedding length:", len(sample_embedding))

null_count = df["embedding"].isnull().sum()
print("Null embeddings:", null_count)

print("\nAll checks passed ✅")

print("\n🎉 Module 6 Completed Successfully!")