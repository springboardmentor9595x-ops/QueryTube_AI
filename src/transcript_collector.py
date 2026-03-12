import pandas as pd
import os
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

def load_metadata():
    progress_path = "data/transcript_progress.csv"
    base_path = "data/cleaned_metadata.csv"

    if os.path.exists(progress_path):
        print("Resuming from existing progress file...")
        df = pd.read_csv(progress_path)
    else:
        print("Loading fresh metadata file...")
        df = pd.read_csv(base_path)

    if "video_id" not in df.columns:
        raise ValueError("video_id column missing in dataset.")
   
    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    return df


def test_single_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        print(f"\nTranscript fetched for video: {video_id}")
        print("Type:", type(transcript))
        print("Number of segments:", len(transcript))

        print("\nSample segments:\n")
        for segment in transcript[:3]:
            print(segment)

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        print(f"Transcript unavailable for {video_id}: {e}")
    except Exception as e:
        print(f"Unexpected error for {video_id}: {e}")


def extract_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        # FIX: remove line breaks so Excel doesn't break rows
        full_text = " ".join(segment.text.replace("\n", " ").strip() for segment in transcript)

        return full_text, None  

    except TranscriptsDisabled:
        return None, "TranscriptsDisabled"

    except NoTranscriptFound:
        return None, "NoTranscriptFound"

    except VideoUnavailable:
        return None, "VideoUnavailable"

    except Exception as e:
        return None, f"OtherError: {str(e)}"


def collect_transcripts(df,
                        save_path="data/transcript_progress.csv",
                        failure_log_path="data/transcript_failures.csv"):

    if "transcript" not in df.columns:
        df["transcript"] = None

    failures = []
    total_videos = len(df)

    for idx, row in df.iterrows():

        if pd.notna(row["transcript"]):
            continue

        video_id = row["video_id"]
        print(f"[{idx+1}/{total_videos}] Fetching transcript for {video_id}")

        transcript_text, failure_reason = extract_transcript(video_id)

        df.at[idx, "transcript"] = transcript_text

        if failure_reason is not None:
            failures.append({
                "video_id": video_id,
                "reason": failure_reason
            })

        # Save progress every 10 videos
        if idx % 10 == 0:
            df.to_csv(save_path, index=False)
            df.to_csv("data/enriched_dataset.csv", index=False)
            print("Progress saved.")

        # Delay to reduce blocking
        time.sleep(random.uniform(2, 4))

    if failures:
        failure_df = pd.DataFrame(failures)
        failure_df.to_csv(failure_log_path, index=False)
        print("Failure log saved.")

    return df


if __name__ == "__main__":
    df = load_metadata()

    print("\nTesting transcript extraction for one video...")
    sample_video_id = df["video_id"].iloc[0]
    test_single_transcript(sample_video_id)

    print("\nStarting transcript collection...\n")
    df = collect_transcripts(df)

    df.to_csv("data/enriched_dataset.csv", index=False)

    print("\nTranscript collection completed.")
    print("Final dataset saved as data/enriched_dataset.csv")