import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------


CHANNEL_ID = "UCzCsyvyrq38R6TnztEzOmgg"

BASE_URL = "https://www.googleapis.com/youtube/v3/search"

# -------------------------------------------------
# STEP 1–5: Extract Data with Pagination
# -------------------------------------------------

all_videos = []
seen_video_ids = set()
next_page_token = None

while True:

    params = {
        "key": API_KEY,
        "channelId": CHANNEL_ID,
        "part": "snippet,id",
        "order": "date",
        "maxResults": 50,
        "pageToken": next_page_token
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print("API Error:", response.status_code)
        print(response.text)
        break

    data = response.json()

    for item in data.get("items", []):

        if item["id"]["kind"] == "youtube#video":

            video_id = item["id"]["videoId"]

            # Avoid duplicates
            if video_id not in seen_video_ids:
                seen_video_ids.add(video_id)

                all_videos.append({
                    "video_id": video_id,
                    "title": item["snippet"]["title"],
                    "publish_date": item["snippet"]["publishedAt"]
                })

    next_page_token = data.get("nextPageToken")

    if not next_page_token:
        break


# -------------------------------------------------
# STEP 7: Convert to DataFrame
# -------------------------------------------------

df = pd.DataFrame(all_videos)

# Ensure correct column order
df = df[["video_id", "title", "publish_date"]]

print("\nDataFrame Created Successfully\n")
print(df.head())


# -------------------------------------------------
# STEP 8: Basic Data Validation
# -------------------------------------------------

print("\n----- DATA VALIDATION -----\n")

print("Total Rows:", len(df))

duplicate_count = df["video_id"].duplicated().sum()
print("Duplicate video IDs:", duplicate_count)

null_values = df.isnull().sum()
print("\nNull Values Per Column:\n", null_values)

print("\nDataset Shape:", df.shape)


# -------------------------------------------------
# STEP 9: Save Dataset as CSV
# -------------------------------------------------

df.to_csv("raw_metadata.csv", index=False)

print("\nCSV file 'raw_metadata.csv' saved successfully.")


