import pandas as pd
import re

df = pd.read_csv("../Module3/video_with_transcripts.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)

# Check required columns
required_cols = ["video_id","title","publish_date","transcript"]

for col in required_cols:
    if col not in df.columns:
        raise Exception(f"Missing column: {col}")

print("All required columns exist")

# STEP 2 — Clean special characters

def clean_text(text):

    if pd.isna(text):
        return text

    text = str(text)

    # remove newline and tab
    text = re.sub(r'\n',' ',text)
    text = re.sub(r'\t',' ',text)

    # remove special characters
    text = re.sub(r'[#@*]','',text)

    # remove multiple spaces
    text = re.sub(r'\s+',' ',text)

    return text.strip()


df["title"] = df["title"].apply(clean_text)

df["transcript"] = df["transcript"].apply(clean_text)

print("Text cleaning completed")

# STEP 3 — Handle missing transcripts

print("Null transcripts before:",df["transcript"].isna().sum())

# Remove rows without transcripts
df = df.dropna(subset=["transcript"])

# Remove empty transcripts
df = df[df["transcript"]!=""]

print("Remaining rows:",len(df))

# STEP 4 — Normalize dataset

# Convert date format
df["publish_date"] = pd.to_datetime(df["publish_date"])

# Rename column to match module format
df = df.rename(columns={
    "publish_date":"datetime"
})

# Keep only required columns
df = df[["video_id","title","datetime","transcript"]]

print("Dataset normalized")

# Save cleaned dataset
df.to_csv("../Module4/cleaned_transcripts.csv",index=False)

print("Saved cleaned_transcripts.csv")

# STEP 5 — Create search queries

queries = [

"What is machine learning",
"Explain neural networks",
"What is artificial intelligence",
"What is deep learning",
"What is supervised learning",
"What is unsupervised learning",
"Explain python programming",
"What is data science",
"What is overfitting",
"What is underfitting",
"What is gradient descent",
"Explain backpropagation",
"What is NLP",
"What is computer vision",
"What is CNN",
"What is RNN",
"What is transformer model",
"What is ChatGPT",
"What is generative AI",
"What is LLM",
"Explain web development",
"What is frontend development",
"What is backend development",
"What is API",
"What is REST API",
"What is database",
"What is SQL"

]

query_df = pd.DataFrame({"query":queries})

query_df.to_csv("../Module4/search_queries.csv",index=False)

print("Search queries saved")

# STEP 6 — Map queries to videos

# Map first queries to first videos

mapping = []

video_ids = df["video_id"].tolist()

for i in range(min(len(queries),len(video_ids))):

    mapping.append({

        "query":queries[i],
        "relevant_video_id":video_ids[i]

    })

mapping_df = pd.DataFrame(mapping)

mapping_df.to_csv("../Module4/query_video_mapping.csv",index=False)

print("Query mapping saved")
