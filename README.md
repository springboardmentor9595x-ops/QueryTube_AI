[# 🎥QueryTube_AI

## 📌 Project Overview
This project is an AI-powered YouTube Semantic Search System that retrieves videos based on meaning (semantic search) rather than exact keyword matching.

## 🚀 Features
- Semantic search using natural language
- Embedding-based retrieval (Sentence Transformers)
- Multiple similarity metrics (Cosine, Euclidean)
- Threshold-based filtering
- Top-K video results
- Gradio UI with embedded YouTube videos

## 🏗️ Workflow
1. Data Collection (YouTube API)
2. Data Cleaning & EDA
3. Transcript Extraction
4. Text Preprocessing
5. Embedding Generation
6. Model Evaluation
7. Video Index Creation
8. Search Optimization
9. UI Deployment (Gradio)
    
## 📂 Project Structure

QueryTube_AI/
│
├── data/ # 📊 Datasets
│ ├── raw_metadata.csv
│ ├── cleaned_metadata.csv
│ ├── video_with_transcripts.csv
│ ├── cleaned_transcripts.csv
│ ├── video_index.csv
│ └── transcript_failures.csv
│
├── notebooks/ # 📒 Jupyter notebooks (optional)
│ └── eda_analysis.ipynb
│
├── scripts/ # ⚙️ Data processing pipeline
│ ├── data_collection.py
│ ├── preprocessing.py
│ ├── transcript_extraction.py
│ ├── embedding.py
│ ├── evaluation.py
│ ├── indexing.py
│
├── src/ # 🧠 Core logic
│ ├── search.py
│ ├── utils.py
│
├── app.py # 🚀 Main Gradio app
├── config.py # ⚙️ Settings
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── LICENSE
## ⚙️ Installation
git clone <repo-url>
cd youtube-semantic-search
pip install -r requirements.txt

## ▶️ Run Project
python app.py

## 🔎 Example
Input: What is overfitting?
Output: Top relevant YouTube videos with scores

## 🧰 Tech Stack
Python, Pandas, NumPy, SentenceTransformers, Gradio, YouTube API

## 📈 Future Work
- Real-time data
- Vector DB (FAISS)
- Advanced UI

## 👨‍💻 Author
AI/NLP Project
](https://github.com/springboardmentor9595x-ops/QueryTube_AI/tree/Lokesh)
