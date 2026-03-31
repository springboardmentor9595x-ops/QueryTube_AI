import pandas as pd
import re


# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------

INPUT_FILE = "video_with_transcripts.csv"
OUTPUT_FILE = "cleaned_transcripts.csv"

print("Loading transcript dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total rows:", len(df))
print(df.columns)


# -------------------------------------------------
# Step 2: Clean Special Characters
# -------------------------------------------------

def clean_text(text):

    if pd.isna(text):
        return text

    text = str(text)

    # remove newline and tab
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # remove unwanted symbols
    text = re.sub(r"[#@*]", "", text)

    # remove punctuation duplicates
    text = re.sub(r"[!?.]{2,}", "", text)

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("Cleaning titles and transcripts...")

df["title"] = df["title"].apply(clean_text)
df["transcript"] = df["transcript"].apply(clean_text)


# -------------------------------------------------
# Step 3: Handle Missing Transcripts
# -------------------------------------------------

print("Handling missing transcripts...")

# Replace empty strings with NaN
df["transcript"] = df["transcript"].replace("", pd.NA)

# OPTION 1: Remove rows without transcripts
df = df.dropna(subset=["transcript"])

# If you prefer placeholder instead, comment above line and use:
# df["transcript"] = df["transcript"].fillna("No transcript available")

print("Remaining rows:", len(df))


# -------------------------------------------------
# Step 4: Normalize Dataset Structure
# -------------------------------------------------

print("Normalizing dataset structure...")


print("Columns before rename:", df.columns)

if "publish_date" in df.columns:
    df = df.rename(columns={"publish_date": "datetime"})

elif "published_at" in df.columns:
    df = df.rename(columns={"published_at": "datetime"})

df = df[["video_id", "title", "datetime", "transcript"]]


# -------------------------------------------------
# Save Cleaned Dataset
# -------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset cleaned successfully")
print("Saved as:", OUTPUT_FILE)