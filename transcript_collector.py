# ------------------------------------------------------------
# YOUTUBE TRANSCRIPT COLLECTION PIPELINE
# Safe version for large datasets (400+ videos)
# ------------------------------------------------------------

import pandas as pd
import time
import random
import os

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

# ------------------------------------------------------------
# File Paths
# ------------------------------------------------------------

INPUT_FILE = "cleaned_metadata.csv"
PROGRESS_FILE = "progress_transcripts.csv"
FINAL_FILE = "video_with_transcripts.csv"
FAILURE_FILE = "transcript_failures.csv"


# ------------------------------------------------------------
# Load Dataset or Resume Progress
# ------------------------------------------------------------

def load_dataset():

    if os.path.exists(PROGRESS_FILE):
        print("Resuming from saved progress...\n")
        df = pd.read_csv(PROGRESS_FILE)

    else:
        print("Starting fresh extraction...\n")
        df = pd.read_csv(INPUT_FILE)

        if "transcript" not in df.columns:
            df["transcript"] = None

    if "video_id" not in df.columns:
        raise ValueError("video_id column missing in dataset")

    # IMPORTANT FIX (prevents pandas dtype error)
    df["transcript"] = df["transcript"].astype("object")

    print("Total videos:", len(df))

    return df


# ------------------------------------------------------------
# Transcript Extraction Function
# ------------------------------------------------------------

def get_transcript(video_id):

    try:
        api = YouTubeTranscriptApi()

        transcript = api.fetch(video_id)

        transcript_data = transcript.to_raw_data()

        transcript_text = " ".join([seg["text"] for seg in transcript_data])

        return transcript_text, None

    except TranscriptsDisabled:
        return None, "Transcripts Disabled"

    except NoTranscriptFound:
        return None, "No Transcript Found"

    except VideoUnavailable:
        return None, "Video Unavailable"

    except Exception as e:

        error_message = str(e)

        # Handle temporary IP block
        if "blocking requests from your IP" in error_message:
            print("\n⚠ IP temporarily blocked. Waiting 5 minutes...\n")
            time.sleep(300)
            return get_transcript(video_id)

        return None, error_message


# ------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------

def main():

    df = load_dataset()

    failures = []

    try:

        for index, row in df.iterrows():

            # Skip already processed videos
            if pd.notna(row["transcript"]):
                continue

            video_id = row["video_id"]

            print(f"Processing {index+1}/{len(df)} : {video_id}")

            transcript, error = get_transcript(video_id)

            if transcript:
                df.at[index, "transcript"] = transcript
            else:
                failures.append({
                    "video_id": video_id,
                    "reason": error
                })

            # Save progress every 10 videos
            if index % 10 == 0:
                df.to_csv(PROGRESS_FILE, index=False)
                print("Progress saved...")

            # Delay to avoid IP blocking
            wait = random.uniform(25, 30)
            print(f"Waiting {round(wait,2)} seconds...\n")
            time.sleep(wait)


    # ------------------------------------------------------------
    # Safe exit if Ctrl+C pressed
    # ------------------------------------------------------------

    except KeyboardInterrupt:

        print("\nCtrl+C detected. Saving progress...")

        df.to_csv(PROGRESS_FILE, index=False)

        if failures:
            pd.DataFrame(failures).to_csv(FAILURE_FILE, index=False)

        print("Progress saved safely. You can resume later.")
        return


    # ------------------------------------------------------------
    # Final Save
    # ------------------------------------------------------------

    df.to_csv(FINAL_FILE, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(FAILURE_FILE, index=False)

    print("\nTranscript extraction completed successfully.")
    print("Final dataset saved as:", FINAL_FILE)


# ------------------------------------------------------------
# Run Pipeline
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\nStarting transcript collection pipeline...\n")

    main()