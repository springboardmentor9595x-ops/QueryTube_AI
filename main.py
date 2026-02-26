import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")

# Constants
BASE_URL = "https://www.googleapis.com/youtube/v3/search"
CHANNEL_ID = "UC7btqG2Ww0_2LwuQxpvo2HQ"

def extract_video_info(item):
    """Extract required fields from each video item"""
    return {
        "video_id": item["id"]["videoId"],
        "title": item["snippet"]["title"],
        "publish_date": item["snippet"]["publishedAt"]
    }

def fetch_all_videos():
    videos = []
    seen_video_ids = set()
    next_page_token = None

    while True:
        params = {
            "key": api_key,
            "channelId": CHANNEL_ID,
            "part": "snippet,id",
            "maxResults": 50,
            "type": "video",
            "order": "date"
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if "items" not in data:
            print("API Error:", data)
            break

        for item in data["items"]:
            video_id = item["id"]["videoId"]

            # Avoid duplicates
            if video_id not in seen_video_ids:
                seen_video_ids.add(video_id)
                videos.append(extract_video_info(item))

        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

    return videos


if __name__ == "__main__":
    all_videos = fetch_all_videos()

    print("Total videos:", len(all_videos))

    df = pd.DataFrame(all_videos)

    df.to_csv("raw_metadata.csv", index=False)

    print("CSV file created successfully ")

