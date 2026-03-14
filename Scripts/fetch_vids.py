import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

CHANNEL_ID = "UC8butISFwT-Wl7EV0hUK0BQ"

CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
PLAYLIST_ITEMS_URL = "https://www.googleapis.com/youtube/v3/playlistItems"

MAX_VIDEOS = 250   # limit number of videos


def get_uploads_playlist_id():
    """Get uploads playlist ID for the channel"""

    params = {
        "key": API_KEY,
        "id": CHANNEL_ID,
        "part": "contentDetails"
    }

    response = requests.get(CHANNELS_URL, params=params)
    data = response.json()

    if "items" not in data:
        raise Exception("Failed to fetch channel data")

    return data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def extract_video_info(item):
    """Extract required metadata fields"""

    snippet = item["snippet"]

    return {
        "video_id": snippet["resourceId"]["videoId"],
        "title": snippet["title"],
        "publish_date": snippet["publishedAt"]
    }


def fetch_videos_limited(uploads_playlist_id):
    """Fetch limited number of videos from uploads playlist"""

    videos = []
    next_page_token = None

    while len(videos) < MAX_VIDEOS:

        params = {
            "key": API_KEY,
            "playlistId": uploads_playlist_id,
            "part": "snippet",
            "maxResults": 50
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(PLAYLIST_ITEMS_URL, params=params)
        data = response.json()

        items = data.get("items", [])

        for item in items:

            videos.append(extract_video_info(item))

            if len(videos) >= MAX_VIDEOS:
                break

        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

    return videos


if __name__ == "__main__":

    uploads_playlist_id = get_uploads_playlist_id()

    print("Uploads Playlist ID:", uploads_playlist_id)

    videos = fetch_videos_limited(uploads_playlist_id)

    print("Total videos fetched:", len(videos))

    df = pd.DataFrame(videos)

    df.to_csv("raw_metadata.csv", index=False)

    print("raw_metadata.csv created successfully")