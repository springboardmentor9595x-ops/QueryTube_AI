# -------------------------------------------------
# MODULE 6: Build Video Index Using Embeddings
# -------------------------------------------------

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

INPUT_FILE = "cleaned_transcripts.csv"
OUTPUT_FILE = "video_index.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

# -------------------------------------------------
# STEP 1: LOAD DATASET
# -------------------------------------------------

def load_dataset():
    try:
        df = pd.read_csv(INPUT_FILE)
        print("Dataset loaded successfully.")
        print("Shape:", df.shape)
        return df
    except Exception as e:
        print("Error loading dataset:", e)
        exit()


# -------------------------------------------------
# STEP 2: VALIDATE DATASET
# -------------------------------------------------

def validate_dataset(df):
    required_columns = ["video_id", "title", "datetime", "transcript"]

    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column: {col}")
            exit()

    print("All required columns are present.")


# -------------------------------------------------
# STEP 3: LOAD MODEL
# -------------------------------------------------

def load_model():
    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully.")
    return model


# -------------------------------------------------
# STEP 4: GENERATE EMBEDDINGS
# -------------------------------------------------

def generate_embeddings(model, df):

    # Handle missing values
    df["title"] = df["title"].fillna("").astype(str)
    df["transcript"] = df["transcript"].fillna("").astype(str)

    print("\nGenerating title embeddings...")
    title_embeddings = model.encode(
        df["title"].tolist(),
        show_progress_bar=True
    )

    print("\nGenerating transcript embeddings...")
    transcript_embeddings = model.encode(
        df["transcript"].tolist(),
        show_progress_bar=True
    )

    print("\nCombining embeddings...")
    combined_embeddings = (title_embeddings + transcript_embeddings) / 2

    print("Embeddings generated successfully.")
    return combined_embeddings


# -------------------------------------------------
# STEP 5: ATTACH EMBEDDINGS TO DATASET
# -------------------------------------------------

def attach_embeddings(df, embeddings):

    embedding_dim = embeddings.shape[1]
    print("Embedding dimension:", embedding_dim)

    embedding_cols = [f"emb_{i}" for i in range(embedding_dim)]

    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)

    df_final = pd.concat([df, embedding_df], axis=1)

    print("Embeddings added to dataset.")
    print("Final shape:", df_final.shape)

    return df_final, embedding_cols


# -------------------------------------------------
# STEP 6: SAVE DATASET
# -------------------------------------------------

def save_dataset(df):
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nVideo index saved as '{OUTPUT_FILE}'")


# -------------------------------------------------
# STEP 7: VALIDATION CHECKS
# -------------------------------------------------

def validate_output(df, embedding_cols):

    print("\nRunning validation checks...")

    # Check nulls
    null_count = df[embedding_cols].isnull().sum().sum()
    print("Null embeddings:", null_count)

    # Check dimension
    print("Embedding columns:", len(embedding_cols))

    # Check rows
    print("Total rows:", len(df))

    print("Validation completed.")


# -------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------

def main():

    print("\n--- MODULE 6: BUILDING VIDEO INDEX ---\n")

    df = load_dataset()
    validate_dataset(df)

    model = load_model()

    embeddings = generate_embeddings(model, df)

    df_final, embedding_cols = attach_embeddings(df, embeddings)

    save_dataset(df_final)

    validate_output(df_final, embedding_cols)

    print("\nProcess completed successfully!")


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

if __name__ == "__main__":
    main()