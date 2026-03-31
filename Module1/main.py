import os
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from datetime import datetime
import pandas as pd

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")


CHANNEL_ID = "UC4SVo0Ue36XCfOyb5Lh1viQ"
BASE_URL = "https://www.googleapis.com/youtube/v3/search"


def fetch_all_videos(channel_id, api_key, max_results=50):

    structured_videos = []
    seen_video_ids = set()
    next_page_token = None

    while True:

        params = {
            "key": api_key,
            "channelId": channel_id,
            "part": "snippet,id",
            "order": "date",
            "maxResults": max_results,
            "type": "video"
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if "error" in data:
            raise Exception(f"API Error: {data['error']}")

        for item in data.get("items", []):

            video_id = item["id"].get("videoId")

            if not video_id or video_id in seen_video_ids:
                continue

            seen_video_ids.add(video_id)

            title = item["snippet"].get("title","").strip()

            published_raw = item["snippet"].get("publishedAt")

            publish_date = None

            if published_raw:
                publish_date = datetime.fromisoformat(
                    published_raw.replace("Z","+00:00")
                ).date()

            structured_videos.append({

                "video_id": video_id,
                "title": title,
                "publish_date": publish_date

            })


        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

    return structured_videos


# Optional transcript function (if needed later)
def fetch_transcript(video_id):

    try:

        api = YouTubeTranscriptApi()

        transcript_list = api.list(video_id)

        transcript = next(iter(transcript_list))

        data = transcript.fetch()

        full_text = " ".join([entry.text for entry in data])

        return full_text

    except Exception as e:

        print(f"Transcript error for {video_id}: {e}")

        return None



if __name__ == "__main__":

    print("Fetching all videos...\n")

    videos = fetch_all_videos(CHANNEL_ID, API_KEY, max_results=50)

    df = pd.DataFrame(videos)

    df = df[["video_id","title","publish_date"]]

    print("DataFrame created successfully.\n")

    print("Total rows:",len(df))

    print("Dataset shape:",df.shape)

    duplicate_count = df["video_id"].duplicated().sum()

    print("Duplicate video_ids:",duplicate_count)

    print("\nNull values per column:")

    print(df.isnull().sum())

    df.to_csv("raw_metadata.csv",index=False)