from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import time

'''# Load the cleaned dataset from cleaned_metadata.csv
df = pd.read_csv('cleaned_metadata.csv') # first instance'''

# Load the cleaned dataset from videos_with_transcripts.csv, transcript_failures.csv

df = pd.read_csv('videos_with_transcripts.csv')
failures = pd.read_csv('transcript_failures.csv')

# Function to fetch and combine transcript text for a single video

def get_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        full_text = " ".join([seg.text for seg in transcript])
        return full_text
    except Exception:
        return None

'''# Apply transcript extraction to all 125 videos with delay

print("Extracting transcripts... (this may take a few minutes)") # first instance
transcripts = []
for i, video_id in enumerate(df['video_id']):
    result = get_transcript(video_id)
    transcripts.append(result)
    print(f"{i+1}/125 done — {video_id}")
    time.sleep(4)  # 4 second delay to avoid YouTube rate limiting

df['transcript'] = transcripts'''

# Apply transcript extraction to the whose transcript didnt fetch yet

print("Extracting transcripts... (this may take a few minutes)")
for i, row in failures.iterrows():
    video_id = row['video_id']
    result = get_transcript(video_id)
    idx = df[df['video_id'] == video_id].index[0]
    df.at[idx, 'transcript'] = result
    print(f"{i+1}/{len(failures)} done — {video_id}")
    time.sleep(30 + (i%10)*2)

# Validation

total          = len(df)
has_transcript = df['transcript'].notna().sum()
missing        = df['transcript'].isna().sum()

print(f"\nTotal videos   : {total}")
print(f"Has transcript : {has_transcript} ({round(has_transcript/total*100, 1)}%)")
print(f"Missing        : {missing} ({round(missing/total*100, 1)}%)")

# Save failures log

failures = df[df['transcript'].isna()][['video_id']].copy()
failures['reason'] = 'transcript unavailable'
failures.to_csv('transcript_failures.csv', index=False)
print("Saved: transcript_failures.csv")

# Save final enriched dataset

df[['video_id', 'title', 'publish_date', 'transcript']].to_csv('videos_with_transcripts.csv', index=False)
print("Saved: videos_with_transcripts.csv")