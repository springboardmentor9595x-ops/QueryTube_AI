import requests
import pandas as pd


import os

API_KEY = os.getenv("API_KEY")
CHANNEL_ID = "UCBwmMxybNva6P_5VmxjzwqA"

BASE_URL = "https://www.googleapis.com/youtube/v3/search"
MAX_PAGES = 10   # control pagination depth


def extract_video_info(item):
    """Extract required fields from each video item"""
    return {
        "video_id": item["id"]["videoId"],
        "title": item["snippet"]["title"],
        "publish_date": item["snippet"]["publishedAt"]
    }


def fetch_videos():
    all_videos = []
    seen_ids = set()
    next_page_token = None
    page_count = 0

    print("Fetching videos using /search endpoint...\n")

    while page_count < MAX_PAGES:

        params = {
            "key": API_KEY,
            "part": "snippet,id",
            "channelId": CHANNEL_ID,
            "type": "video",
            "order": "date",
            "maxResults": 50
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if "items" not in data:
            print("API Error:", data)
            break

        print(f"Page {page_count + 1} results:", len(data["items"]))

        for item in data["items"]:
            video_id = item["id"]["videoId"]

            if video_id not in seen_ids:
                seen_ids.add(video_id)
                all_videos.append(extract_video_info(item))

        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            print("No more pages available.")
            break

        page_count += 1

    return all_videos


def validate_dataframe(df):
    print("\nData Validation")
    print("Shape:", df.shape)
    print("Duplicate IDs:", df["video_id"].duplicated().sum())
    print("Null values:\n", df.isnull().sum())


if __name__ == "__main__":

    videos = fetch_videos()

    print("\nTotal videos fetched:", len(videos))

    df = pd.DataFrame(videos)
    df["publish_date"] = pd.to_datetime(df["publish_date"])

    validate_dataframe(df)

    df.to_csv("raw_metadata.csv", index=False)

    print("\nDataset saved as raw_metadata.csv")