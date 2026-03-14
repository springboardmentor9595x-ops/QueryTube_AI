from sentence_transformers import SentenceTransformer
import pandas as pd
import re

# Load Dataset

df = pd.read_csv('videos_with_transcripts.csv')
print(f"Loaded: {len(df)} rows")

# Clean Special Characters

def clean_text(text):
    if pd.isna(text):
        return text
    text = re.sub(r'[\n\t#@*]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df['title'] = df['title'].apply(clean_text)
df['transcript'] = df['transcript'].apply(clean_text)
print("Cleaning done!")

# Handle Missing Transcripts

before = len(df)
df = df.dropna(subset=['transcript'])
df = df[df['transcript'].str.strip() != '']
after = len(df)
print(f"Removed {before - after} rows with missing transcripts")

# Normalize Dataset Format

df = df.rename(columns={'publish_date': 'datetime'})
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[['video_id', 'title', 'datetime', 'transcript']]
print(f"Final dataset: {len(df)} rows")

df.to_csv('cleaned_transcripts.csv', index=False)
print("Saved: cleaned_transcripts.csv")

# Search Queries and Mapping

query_video_map = [
    {"query": "What shape are raindrops?", "relevant_video_id": "AjeTIujh0gE"},
    {"query": "Why do bugs die on their backs?", "relevant_video_id": "nEXIC62dOLY"},
    {"query": "What is the fastest animal on Earth?", "relevant_video_id": "8HmXeO8aO4g"},
    {"query": "What is the most venomous animal?", "relevant_video_id": "B1NLD828sYI"},
    {"query": "Why do birds fly in a V formation?", "relevant_video_id": "XYxLL5ehYzc"},
    {"query": "What is the sleepiest animal?", "relevant_video_id": "Kd4GreqBPJI"},
    {"query": "What is the deepest living thing?", "relevant_video_id": "y3KPk5lJJyE"},
    {"query": "What is the heaviest living organism?", "relevant_video_id": "mLA1bs_CuFw"},
    {"query": "What is the loudest animal on Earth?", "relevant_video_id": "ooeIJGmteO4"},
    {"query": "What is the most invincible creature?", "relevant_video_id": "h-UjcELwzAk"},
    {"query": "How do insects fly in the rain?", "relevant_video_id": "CfbY_lYf0Yk"},
    {"query": "What are banana strings?", "relevant_video_id": "21TKk1fLkKQ"},
    {"query": "Why do we see faces in things?", "relevant_video_id": "L4zFS0RPP7c"},
    {"query": "Can humans hibernate?", "relevant_video_id": "_0wQxH9naaM"},
    {"query": "What animal lives the shortest life?", "relevant_video_id": "CcljLB6qRwI"},
    {"query": "How do scorpions survive?", "relevant_video_id": "HjsaNE0Mtco"},
    {"query": "What are scallop eyes?", "relevant_video_id": "i0b_ci7RoZE"},
    {"query": "How did feathered dinosaurs relate to flight?", "relevant_video_id": "yLB839Sir5c"},
    {"query": "What is de-extinction?", "relevant_video_id": "mJx2h80UGj4"},
    {"query": "Could AI help us talk to whales?", "relevant_video_id": "Ogr9kbypSNg"},
    {"query": "What is the blackest animal on Earth?", "relevant_video_id": "ZDSdQZPgzkg"},
    {"query": "How do you tag a blue whale?", "relevant_video_id": "YG3-Pw3rixU"},
    {"query": "Why are starfish different than we thought?", "relevant_video_id": "xIDhPocxWAA"},
    {"query": "What animal made footprints in North America?", "relevant_video_id": "xDP3JQkJSAo"},
    {"query": "Why is snow so quiet?", "relevant_video_id": "mT285PhyI7E"},
    {"query": "What happens when matter goes faster than light?", "relevant_video_id": "Cuf_9l4fQDA"},
    {"query": "How do cats always land on their feet?", "relevant_video_id": "zDzfmPhbE3U"},
    {"query": "Why does spaghetti break in three pieces?", "relevant_video_id": "fNtzwdFR1EY"},
    {"query": "What is the physics of falling toast?", "relevant_video_id": "NPnaac9Qq9A"},
    {"query": "How do you steer at the speed of light?", "relevant_video_id": "ZL6Wil_8OpA"},
    {"query": "How do air conditioners make Earth warmer?", "relevant_video_id": "42orkHXmLdI"},
    {"query": "Can an apple defy gravity?", "relevant_video_id": "T52bUKPs0Bg"},
    {"query": "What is the hottest thing ever made?", "relevant_video_id": "9Fd3xnS_Hl4"},
    {"query": "How do you make fire by smashing balls?", "relevant_video_id": "OvM5zeYL7fU"},
    {"query": "Why does sunscreen look weird in UV light?", "relevant_video_id": "wlACYsTEayA"},
    {"query": "What time is it on Voyager 1?", "relevant_video_id": "UI8pp2j2fZ8"},
    {"query": "What could we see with a planet-sized telescope?", "relevant_video_id": "9tI7V4hYHaI"},
    {"query": "Why are solar eclipses a big deal?", "relevant_video_id": "791qJZivHpk"},
    {"query": "How do we measure the size of the universe?", "relevant_video_id": "R451yqCHoc0"},
    {"query": "What is dark energy?", "relevant_video_id": "EoVQrsmnKC8"},
    {"query": "How do we clean up space junk?", "relevant_video_id": "uJcXCdbm77g"},
    {"query": "Why did NASA punch an asteroid?", "relevant_video_id": "tp9BQ88rNso"},
    {"query": "What is the worst idea in space history?", "relevant_video_id": "BTbEnmSu-nY"},
    {"query": "What is Earth's new space friend?", "relevant_video_id": "SZpEb0WImSU"},
    {"query": "What does space sound like?", "relevant_video_id": "fOMEwrilIdw"},
    {"query": "What is the biggest camera ever made?", "relevant_video_id": "WMLuYdo1kUQ"},
    {"query": "How to go faster than light speed?", "relevant_video_id": "akBpQ-A7mCQ"},
    {"query": "Where does lost weight go?", "relevant_video_id": "hyarxb96lgM"},
    {"query": "Why is pee yellow?", "relevant_video_id": "Pwid3u6mIqk"},
    {"query": "What makes some people taste colors?", "relevant_video_id": "pPIem63bC4w"},
    {"query": "Why doesn't a heart look like a heart?", "relevant_video_id": "tygW78iVpsg"},
    {"query": "What is the law of urination?", "relevant_video_id": "dapX-TAIfDY"},
    {"query": "Why do some people hear differently?", "relevant_video_id": "skUl27-VwpE"},
    {"query": "What is the most detailed brain map?", "relevant_video_id": "YSM13kIpV4Q"},
    {"query": "Why is grandmother an evolutionary mystery?", "relevant_video_id": "iKLwzmjfcW4"},
    {"query": "What is wrong with what we learned about genetics?", "relevant_video_id": "fh9xTJQSFmE"},
    {"query": "How did custom gene editing save a baby?", "relevant_video_id": "375IgtuZmnQ"},
    {"query": "Why is autism diagnosis increasing?", "relevant_video_id": "E-yaxqDsfgY"},
    {"query": "How do blood types work?", "relevant_video_id": "vit6p0-ovqo"},
    {"query": "What is the sexy paradox that stumped Darwin?", "relevant_video_id": "xEJrlQYkuVo"},
    {"query": "How does butterfly metamorphosis really work?", "relevant_video_id": "4RaCURU6A2o"},
    {"query": "Are spidergoats real?", "relevant_video_id": "GkumBTzW9HY"},
    {"query": "Why is the ocean salty?", "relevant_video_id": "HrYU-RSprrg"},
    {"query": "Where are the oldest rocks on Earth?", "relevant_video_id": "c4GRLFuTyQY"},
    {"query": "What happened to Mars?", "relevant_video_id": "ABaLGn2UdpE"},
    {"query": "What is the flaw in photosynthesis?", "relevant_video_id": "DZ_T4zMBx6E"},
    {"query": "How bad was pollution in the past?", "relevant_video_id": "Ssmkpys6n8A"},
    {"query": "How is climate change affecting sea turtles?", "relevant_video_id": "cKgAiWVnk6E"},
    {"query": "Can superplants fight climate change?", "relevant_video_id": "22GuteWAVCk"},
    {"query": "Can oysters save New York City?", "relevant_video_id": "O_mk-YJxwLw"},
    {"query": "Why are coral reefs in trouble?", "relevant_video_id": "x0qavLcoI0g"},
    {"query": "What is the weirdest fruit on Earth?", "relevant_video_id": "fzT2ZfGsfVQ"},
    {"query": "Where does dust come from?", "relevant_video_id": "2KaAHD2TOos"},
    {"query": "Why is useless knowledge useful?", "relevant_video_id": "KELjtarJ2TY"},
    {"query": "How are real diamonds made in a lab?", "relevant_video_id": "8ieWJq0MaXw"},
    {"query": "What is camouflage really?", "relevant_video_id": "qiuPCIqstOo"},
    {"query": "How did X come to mean unknown?", "relevant_video_id": "Bq_ZgV4OURI"},
    {"query": "Where do new ideas come from?", "relevant_video_id": "thtKslF8zE4"},
    {"query": "Why is ROYGBIV wrong?", "relevant_video_id": "PjUxZ5pWJ-g"},
    {"query": "What is missing from the rainbow?", "relevant_video_id": "gVZwdYZqCUI"},
    {"query": "What is the randomness crisis on the internet?", "relevant_video_id": "xLeiJA2RuNc"},
    {"query": "How do we predict death?", "relevant_video_id": "9s3fgnkNkaA"},
    {"query": "Why do cicadas emerge in cycles?", "relevant_video_id": "bAe5Tg-EIxk"},
]

queries_df = pd.DataFrame(query_video_map)
queries_df.to_csv('search_queries.csv', index=False)
print(f"Saved: search_queries.csv ({len(queries_df)} queries)")

# Sentence Transformer Model Exploration

print("\nLoading model: multi-qa-MiniLM-L6-cos-v1")
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
sample = df['transcript'].tolist()[:3]
embeddings = model.encode(sample, show_progress_bar=True)
print(f"Embedding shape: {embeddings.shape}")
print("Model exploration done!")