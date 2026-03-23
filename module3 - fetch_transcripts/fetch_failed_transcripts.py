from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import time
import random

DATA_FILE = "video_with_transcripts.csv"
RETRY_BATCH = 10   # number of failed videos per batch

print("Loading dataset...\n")

data = pd.read_csv(DATA_FILE)

api_client = YouTubeTranscriptApi()

failed_subset = data[data["status"] == "failed"]

total_failed = len(failed_subset)
total_records = len(data)

print(f"Total videos in dataset: {total_records}")
print(f"Failed videos to retry: {total_failed}\n")


def retrieve_transcript(v_id):

    try:
        transcript_collection = api_client.list(v_id)

        try:
            chosen = transcript_collection.find_transcript(['en'])
        except:
            try:
                chosen = transcript_collection.find_generated_transcript(['en'])
            except:
                chosen = list(transcript_collection)[0]

        entries = chosen.fetch()

        transcript_text = " ".join([item.text for item in entries])

        return transcript_text

    except Exception as error:
        print("Error:", str(error)[:120])
        return None


batch_counter = 0

for position, row_index in enumerate(failed_subset.index, start=1):

    vid_id = data.loc[row_index, "video_id"]

    print(f"\nRetrying ({position}/{total_failed}) : {vid_id}")

    text_result = retrieve_transcript(vid_id)

    if text_result:
        data.loc[row_index, "transcript"] = text_result
        data.loc[row_index, "status"] = "done"
        print("Transcript recovered")
    else:
        print("Still unavailable")

    batch_counter += 1

    delay = random.randint(6,12)
    print(f"Waiting {delay} seconds...")
    time.sleep(delay)

    if batch_counter >= RETRY_BATCH:

        print("\nBatch retry completed")

        data.to_csv(DATA_FILE, index=False)
        print("CSV saved successfully")

        done_total = (data["status"] == "done").sum()
        fail_total = (data["status"] == "failed").sum()

        progress = (done_total / total_records) * 100

        print("\nCurrent Progress")
        print("----------------------")
        print(f"Done videos: {done_total}")
        print(f"Failed videos: {fail_total}")
        print(f"Completion: {progress:.2f}%")

        print("\nOptions:")
        print("ENTER → Continue retrying")
        print("Q → Quit safely")

        choice = input("\nYour choice: ").lower().strip()

        if choice == "q":
            print("\nStopping retry process safely.")
            break

        batch_counter = 0

        cooldown = random.randint(60,120)
        print(f"\nCooling down for {cooldown} seconds...\n")
        time.sleep(cooldown)

# Final save
data.to_csv(DATA_FILE, index=False)

done_total = (data["status"] == "done").sum()
fail_total = (data["status"] == "failed").sum()

progress = (done_total / total_records) * 100

print("\nFinal Progress")
print("----------------------")
print(f"Done videos: {done_total}")
print(f"Failed videos: {fail_total}")
print(f"Completion: {progress:.2f}%")

print("\nRetry process finished.")