import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("Youtube_API")
CHANNEL_ID = "UCKgpamMlm872zkGDcBJHYDg"

search_url = "https://www.googleapis.com/youtube/v3/search"

all_videos = []
next_page_token = None

print("Fetching videos with pagination...\n")

while True:
    params = {
        "part": "snippet",
        "channelId": CHANNEL_ID,
        "maxResults": 50,
        "order": "date",
        "type": "video",
        "key": API_KEY
    }

    if next_page_token:
        params["pageToken"] = next_page_token

    response = requests.get(search_url, params=params)

    if response.status_code != 200:
        print("Error:", response.status_code)
        print(response.text)
        break

    data = response.json()

    for item in data.get("items", []):
        video_id = item["id"].get("videoId")
        title = item["snippet"].get("title")
        publish_date = item["snippet"].get("publishedAt")

        if video_id:  # avoid None
            all_videos.append({
                "video_id": video_id,
                "title": title,
                "publish_date": publish_date
            })

    next_page_token = data.get("nextPageToken")

    if not next_page_token:
        break

print(f"\nTotal videos fetched (before dedupe): {len(all_videos)}")


# ---------------------------------
# Remove duplicate video IDs
# ---------------------------------

unique_videos = {video["video_id"]: video for video in all_videos}
clean_data = list(unique_videos.values())

print(f"Total videos after removing duplicates: {len(clean_data)}")


# ---------------------------------
# Convert to DataFrame
# ---------------------------------

df = pd.DataFrame(clean_data)

# Rename columns clearly
df = df[["video_id", "title", "publish_date"]]


# ---------------------------------
# Basic Data Validation
# ---------------------------------

print("\nData Validation:")
print("Dataset Shape:", df.shape)
print("Duplicate video IDs:", df["video_id"].duplicated().sum())
print("Null Values:\n", df.isnull().sum())


# ---------------------------------
# Save to CSV
# ---------------------------------

df.to_csv("raw_metadata.csv", index=False)

print("\nDataset saved as raw_metadata.csv successfully.")