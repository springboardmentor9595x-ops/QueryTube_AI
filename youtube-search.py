import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID")

# YouTube API endpoint

BASE_URL = "https://www.googleapis.com/youtube/v3/search"

# Parameters

def fetch_page(page_token=None):
    params = {
        'part': 'snippet',
        'channelId': CHANNEL_ID,
        'maxResults': 50,
        'order': 'date',
        'type': 'video',
        'key': API_KEY
    }
    if page_token:
        params['pageToken'] = page_token

    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None

    return response.json()

# Pagination — loop through ALL pages

def fetch_all_videos():
    all_items = []
    page_token = None
    page_num = 1

    while True:
        print(f"Fetching page {page_num}...")
        data = fetch_page(page_token)

        if data is None:
            break

        items = data.get('items', [])
        all_items.extend(items)
        print(f"-> {len(items)} videos | Running total: {len(all_items)}")

        # Get next page token — if missing, we're done
        page_token = data.get('nextPageToken')
        if not page_token:
            print("All pages collected.\n")
            break

        page_num += 1

    return all_items

# Extract only the 3 required fields 

def extract_fields(items):
    videos = []
    seen_ids = set()  # Prevents duplicate entries across pages

    for item in items:
        # Skip anything that isn't a video (safety check)
        if item.get('id', {}).get('kind') != 'youtube#video':
            continue

        video_id = item['id']['videoId']

        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)

        snippet = item.get('snippet', {})

        videos.append({
            'video_id': video_id,
            'title': snippet.get('title', '').strip(),
            'publish_date': snippet.get('publishedAt', '')[:10]  # YYYY-MM-DD only
        })

    return videos

# Build Pandas Dataframe

def build_dataframe(videos):
    df = pd.DataFrame(videos, columns=['video_id', 'title', 'publish_date'])
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df.sort_values('publish_date', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Data Validation

def validate(df):
    print("=" * 40)
    print("  VALIDATION REPORT")
    print("=" * 40)
    print(f"  Shape          : {df.shape}")
    print(f"  Duplicate IDs  : {df['video_id'].duplicated().sum()}")
    print(f"  Null values    : {df.isnull().sum().sum()}")
    print(f"  Date range     : {df['publish_date'].min().date()}  ->  {df['publish_date'].max().date()}")
    print("=" * 40)

    assert df['video_id'].duplicated().sum() == 0, "Duplicate video IDs found!"
    assert df.isnull().sum().sum() == 0,            "Null values detected!"
    print("  All checks passed.\n")

# Save to CSV

def save(df, filename="raw_metadata.csv"):
    df.to_csv(filename, index=False)
    print(f"Saved -> {filename}")

# RUN

if __name__ == "__main__":
    print("Starting video collection...\n")

    raw_items = fetch_all_videos()
    videos = extract_fields(raw_items)
    df = build_dataframe(videos)

    validate(df)

    print("Preview (first 5 rows):")
    print(df.head().to_string(index=False))
    print()

    save(df)