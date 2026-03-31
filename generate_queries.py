import pandas as pd
from keybert import KeyBERT
import random

print("Loading transcript dataset...")

df = pd.read_csv("cleaned_transcripts.csv")

print("Total videos:", len(df))

# Initialize keyword model
kw_model = KeyBERT()

queries = []

print("Generating queries from transcripts...")

for index, row in df.iterrows():

    transcript = str(row["transcript"])

    if len(transcript) < 50:
        continue

    keywords = kw_model.extract_keywords(
        transcript,
        keyphrase_ngram_range=(1,2),
        stop_words="english",
        top_n=2
    )

    for kw, score in keywords:

        queries.append(f"What is {kw}?")
        queries.append(f"Explain {kw}")
        queries.append(f"How does {kw} work?")

print("Total queries generated:", len(queries))

# shuffle queries
random.shuffle(queries)

# keep only 80 queries
queries = queries[:80]

query_df = pd.DataFrame({"query": queries})

query_df.to_csv("search_queries.csv", index=False)

print("Queries saved as search_queries.csv")