from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import os
import time
import random

INPUT_FILE = "../Module2/cleaned_metadata.csv"
OUTPUT_FILE = "../Module3/video_with_transcripts.csv"

BATCH_SIZE = 15   # videos per batch

print("\nLoading dataset...\n")

# Resume if output already exists
if os.path.exists(OUTPUT_FILE):
    print("Resuming from existing file...\n")
    df = pd.read_csv(OUTPUT_FILE)
else:
    df = pd.read_csv(INPUT_FILE)
    df["transcript"] = None
    df["status"] = None

yt_api = YouTubeTranscriptApi()

total = len(df)

print(f"Total videos: {total}")
print("Starting transcript extraction...\n")


def fetch_transcript(video_id):

    try:
        transcript_list = yt_api.list(video_id)

        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
            except:
                transcript = list(transcript_list)[0]

        fetched = transcript.fetch()

        text = " ".join([x.text for x in fetched])

        return text

    except Exception as e:
        print("Error:", str(e)[:120])
        return None


processed_in_batch = 0

for i in range(total):

    # Skip already processed rows
    if pd.notna(df.loc[i, "status"]):
        continue

    video_id = df.loc[i, "video_id"]

    print(f"\nProcessing {i+1}/{total} : {video_id}")

    transcript = fetch_transcript(video_id)

    if transcript:
        df.loc[i, "transcript"] = transcript
        df.loc[i, "status"] = "done"
        print("Transcript extracted")
    else:
        df.loc[i, "status"] = "failed"
        print("Transcript failed")

    processed_in_batch += 1

    # small random delay (avoids blocking)
    delay = random.randint(5,10)
    print(f"Waiting {delay} seconds...")
    time.sleep(delay)

    # If batch completed
    if processed_in_batch >= BATCH_SIZE:

        print("\nBatch completed")

        # Save CSV
        df.to_csv(OUTPUT_FILE, index=False)

        print("CSV updated successfully")
        print("You can open the file and verify transcripts\n")

        done_count = len(df[df["status"] == "done"])
        failed_count = len(df[df["status"] == "failed"])

        print("Current Stats")
        print("Done:", done_count)
        print("Failed:", failed_count)
        print("Remaining:", total - (done_count + failed_count))

        print("\nOptions:")
        print("ENTER → Continue next batch")
        print("Q → Quit safely")

        user = input("\nYour choice: ").lower().strip()

        if user == "q":
            print("\nStopping safely.")
            print("You can resume later by running the script again.")
            break

        processed_in_batch = 0

        cooldown = random.randint(60,120)
        print(f"\nCooling down {cooldown} seconds before next batch...\n")
        time.sleep(cooldown)

# Final save
df.to_csv(OUTPUT_FILE, index=False)

print("\nExtraction session finished.")
print("File saved as:", OUTPUT_FILE)
