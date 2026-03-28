# ================================
# MODULE 6: VIDEO INDEX BUILDING
# FINAL VERSION (CLEAN)
# ================================

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

# ================================
# STEP 1: LOAD DATASET
# ================================

print("\n🔹 Loading dataset...\n")

file_path = "cleaned_transcripts.csv"
df = pd.read_csv(file_path)

required_columns = ['video_id', 'title', 'datetime', 'transcript']

# Validate columns
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns: {missing_cols}")

print("✅ Dataset loaded successfully")
print("Shape:", df.shape)

# Fill missing values
df['title'] = df['title'].fillna("")
df['transcript'] = df['transcript'].fillna("")

# ================================
# STEP 2: LOAD MODEL
# ================================

print("\n🔹 Loading embedding model...\n")

model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ Model loaded successfully")

# ================================
# STEP 3: GENERATE TITLE EMBEDDINGS
# ================================

print("\n🔹 Generating title embeddings...\n")

title_embeddings = model.encode(
    df['title'].tolist(),
    show_progress_bar=True,
    batch_size=32
)

print("✅ Title embeddings shape:", np.array(title_embeddings).shape)

# ================================
# STEP 4: GENERATE TRANSCRIPT EMBEDDINGS
# ================================

print("\n🔹 Generating transcript embeddings...\n")

transcript_embeddings = model.encode(
    df['transcript'].tolist(),
    show_progress_bar=True,
    batch_size=32
)

print("✅ Transcript embeddings shape:", np.array(transcript_embeddings).shape)

# ================================
# STEP 5: COMBINE EMBEDDINGS
# ================================

print("\n🔹 Combining embeddings (average)...\n")

combined_embeddings = (title_embeddings + transcript_embeddings) / 2

print("✅ Combined embeddings shape:", combined_embeddings.shape)

# ================================
# STEP 6: ATTACH EMBEDDINGS
# ================================

print("\n🔹 Attaching embeddings to dataset...\n")

embedding_df = pd.DataFrame(combined_embeddings)

# Rename columns
embedding_dim = embedding_df.shape[1]
embedding_df.columns = [f'embedding_{i}' for i in range(embedding_dim)]

# Merge
final_df = pd.concat([df, embedding_df], axis=1)

print("✅ Final dataset shape:", final_df.shape)

# ================================
# STEP 7: SAVE CSV ONLY
# ================================

print("\n🔹 Saving dataset...\n")

output_file = "video_index.csv"
final_df.to_csv(output_file, index=False)

print(f"✅ File saved: {output_file}")

# ================================
# STEP 8: VALIDATION (STEP 9)
# ================================

print("\n🔹 Running validation checks...\n")

# 1. Row count check
if len(final_df) == len(df):
    print("✅ Row count matches original dataset")
else:
    print("❌ Row count mismatch!")

# 2. Embedding existence check
if embedding_df.isnull().sum().sum() == 0:
    print("✅ No missing embeddings")
else:
    print("❌ Missing embeddings detected!")

# 3. Embedding dimension consistency
expected_dim = len(combined_embeddings[0])
actual_dim = embedding_df.shape[1]

if expected_dim == actual_dim:
    print(f"✅ Embedding dimension consistent: {actual_dim}")
else:
    print("❌ Embedding dimension mismatch!")

# 4. Final null check
total_nulls = final_df.isnull().sum().sum()
if total_nulls == 0:
    print("✅ No null values in final dataset")
else:
    print(f"⚠️ Found {total_nulls} null values")

# 5. Sample check
print("\n🔍 Sample data preview:")
print(final_df.head(2))

# ================================
# DONE
# ================================

print("\n🎉 MODULE 6 COMPLETED SUCCESSFULLY!")