import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
CHANNEL_ID = "UC8butISFwT-Wl7EV0hUK0BQ"

MAX_PAGES = 20


def get_upload_playlist():
    """
    Get the uploads playlist ID of the channel
    """

    url = "https://www.googleapis.com/youtube/v3/channels"

    params = {
        "key": API_KEY,
        "id": CHANNEL_ID,
        "part": "contentDetails"
    }

    response = requests.get(url, params=params).json()

    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def extract_video_info(item):
    return {
        "video_id": item["snippet"]["resourceId"]["videoId"],
        "title": item["snippet"]["title"],
        "publish_date": item["snippet"]["publishedAt"]
    }


def fetch_videos(playlist_id):

    url = "https://www.googleapis.com/youtube/v3/playlistItems"

    all_videos = []
    next_page_token = None
    page = 0

    print("Fetching videos from Uploads Playlist...\n")

    while page < MAX_PAGES:

        params = {
            "key": API_KEY,
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": 50
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(url, params=params).json()

        items = response.get("items", [])

        print(f"Page {page+1}: {len(items)} videos")

        for item in items:
            all_videos.append(extract_video_info(item))

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

        page += 1

    return all_videos


def validate_dataframe(df):

    print("\nData Validation")
    print("Shape:", df.shape)
    print("Duplicate IDs:", df["video_id"].duplicated().sum())
    print("Null values:\n", df.isnull().sum())


if __name__ == "__main__":
    

    playlist_id = get_upload_playlist()

    print("Uploads Playlist ID:", playlist_id)

    videos = fetch_videos(playlist_id)

    print("\nTotal videos fetched:", len(videos))

    df = pd.DataFrame(videos)

    if not df.empty:
        df["publish_date"] = pd.to_datetime(df["publish_date"])
        validate_dataframe(df)

        df.to_csv("raw_metadata.csv", index=False)

        print("\nDataset saved as raw_metadata.csv")