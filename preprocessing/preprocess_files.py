import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
from tqdm import tqdm
import textstat
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from openai import OpenAI
import streamlit as st

# Global models
analyzer = SentimentIntensityAnalyzer()
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,
    model_kwargs={"torch_dtype": torch.float32}
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def is_app_preprocessed(app_id, data_dir="data"):
    """Check if all expected output files already exist for the app."""
    app_path = os.path.join(data_dir, app_id)
    expected_files = [
        f"{app_id}_reviews_preprocessed.csv",
        "topics.csv",
        "emotion_distribution_by_topic.csv",
        "pain_point_summary.csv"
    ]
    return all(os.path.exists(os.path.join(app_path, f)) for f in expected_files)

def batch_emotion_labels(texts, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Emotion Classification"):
        batch = texts[i:i+batch_size]
        try:
            batch_preds = emotion_pipeline(batch, truncation=True)
            for result in batch_preds:
                if isinstance(result, list):
                    results.append(result[0]['label'])
                else:
                    results.append(result['label'])
        except Exception as e:
            print(f"Error on batch {i}: {e}")
            results.extend(["error"] * len(batch))
    return results

def suggest_label(topic_num, top_words, sample_reviews):
    top_words_str = ', '.join(top_words)
    prompt = f"""
You are analyzing app reviews that have been grouped into topics. 
Your job is to give a concise label (2-5 words max) that summarizes the theme of the topic.

Here are the top words for Topic {topic_num}:
{top_words_str}

Here are 3 representative reviews:
1. {sample_reviews[0]}
2. {sample_reviews[1]}
3. {sample_reviews[2]}

What is the best label for this topic?
Respond with only the label.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error labeling topic {topic_num}: {e}")
        return "Unknown"

def consolidate_merged_topics(df, max_clusters=10, top_n_words=10):
    unique_topics = sorted(df['merged_topic'].dropna().unique())
    n_clusters = min(max_clusters, len(unique_topics))
    if n_clusters == 0:
        return df  # nothing to cluster

    def get_top_words_for_topic_label(label):
        texts = df[df['merged_topic'] == label]['text'].dropna().tolist()
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        word_counts = Counter(all_words)
        top_words = [word for word, count in word_counts.most_common(top_n_words)]
        return " ".join(top_words) if top_words else "no words"

    topic_texts = [get_top_words_for_topic_label(t) for t in unique_topics]
    embeddings = embedding_model.encode(topic_texts, show_progress_bar=False)

    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(embeddings)

    topic_to_cluster = dict(zip(unique_topics, cluster_labels))

    cluster_to_label = {}
    for cluster in range(n_clusters):
        member_topics = [t for t, c in topic_to_cluster.items() if c == cluster]
        labels = df[df['merged_topic'].isin(member_topics)]['merged_topic_label'].unique()
        cluster_to_label[cluster] = labels[0] if len(labels) > 0 else f"Cluster {cluster}"

    df['merged_topic'] = df['merged_topic'].map(topic_to_cluster)
    df['merged_topic_label'] = df['merged_topic'].map(cluster_to_label)

    return df

def clean_review_text(text):
    # Simple cleaning — adjust if you want more
    return text.strip()

def load_reviews_file(path):
    return pd.read_csv(path)

def run_topic_modeling(texts):
    topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
    topics, probs = topic_model.fit_transform(texts)
    return topics, probs, topic_model

def analyze_emotions(df):
    df['emotion'] = batch_emotion_labels(df['text'].tolist())
    return df

def analyze_sentiment(df):
    df['vader_sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

def analyze_readability(df):
    df['readability'] = df['text'].apply(textstat.flesch_reading_ease)
    df['text_length'] = df['text'].str.len()
    return df

def detect_keywords(df):
    bug_keywords = ['bug', 'crash', 'glitch', 'error', 'issue', 'broken', 'freeze']
    request_keywords = ['add', 'feature', 'please', 'would like', 'suggest', 'can you', 'request']
    df['contains_bug_keywords'] = df['text'].str.lower().apply(lambda x: any(k in x for k in bug_keywords))
    df['contains_request_keywords'] = df['text'].str.lower().apply(lambda x: any(k in x for k in request_keywords))
    return df

def process_reviews_file(file_path, app_folder, app_id):
    print(f"Processing {file_path}...")
    df = load_reviews_file(file_path)
    df["text"] = df["text"].astype(str).apply(clean_review_text)
    df = df[df["text"].str.strip().astype(bool)]

    df = analyze_emotions(df)
    df = analyze_sentiment(df)
    df = analyze_readability(df)
    df = detect_keywords(df)

    topics, probs, topic_model = run_topic_modeling(df["text"].tolist())
    df["topic"] = topics

    unique_topics = sorted(df["topic"].unique())
    topic_representative = {}
    for topic in unique_topics:
        texts = df[df["topic"] == topic]["text"].tolist()
        sample_reviews = texts[:3] if len(texts) >= 3 else texts + [""]*(3-len(texts))
        top_words = [word for word, _ in topic_model.get_topic(topic)[:10]] if topic != -1 else []
        topic_representative[topic] = (top_words, sample_reviews)

    topic_labels = {}
    for topic_num, (top_words, sample_reviews) in tqdm(topic_representative.items(), desc="GPT Topic Labeling"):
        if topic_num == -1:
            topic_labels[topic_num] = "Other"
        else:
            label = suggest_label(topic_num, top_words, sample_reviews)
            topic_labels[topic_num] = label

    df["merged_topic"] = df["topic"]
    df["merged_topic_label"] = df["topic"].map(topic_labels)

    df = consolidate_merged_topics(df, max_clusters=10)

    negative_emotions = ['anger', 'sadness', 'fear', 'disgust']
    topic_counts = df.groupby('merged_topic_label').size()
    neg_counts = df[df['emotion'].isin(negative_emotions)].groupby('merged_topic_label').size()
    neg_proportion = (neg_counts / topic_counts).fillna(0)
    threshold = 0.4
    pain_points = neg_proportion[neg_proportion > threshold]
    df['pain_point'] = df['merged_topic_label'].isin(pain_points.index)

    output_path = os.path.join(app_folder, f"{app_id}_reviews_preprocessed.csv")
    df.to_csv(output_path, index=False)

    topic_summary = df.groupby("merged_topic_label").agg({
        "text": "count",
        "vader_sentiment": "mean",
        "text_length": "mean",
        "readability": "mean",
        "contains_bug_keywords": "mean",
        "contains_request_keywords": "mean"
    }).rename(columns={
        "text": "total_reviews",
        "vader_sentiment": "avg_vader_sentiment",
        "text_length": "avg_text_length",
        "readability": "avg_readability",
        "contains_bug_keywords": "bug_keyword_rate",
        "contains_request_keywords": "request_keyword_rate"
    }).reset_index()
    topic_summary["pain_point_flag"] = topic_summary["merged_topic_label"].isin(pain_points.index)
    topic_summary.to_csv(os.path.join(app_folder, "topics.csv"), index=False)

    pain_summary = pd.DataFrame({
        'negative_proportion': neg_proportion,
        'total_reviews': topic_counts,
        'pain_point_flag': neg_proportion > threshold
    }).reset_index()
    pain_summary.to_csv(os.path.join(app_folder, "pain_point_summary.csv"), index=False)

    emotion_dist = (
        df[["merged_topic_label", "emotion"]]
        .value_counts(normalize=True)
        .reset_index(name="proportion")
    )
    pivoted = emotion_dist.pivot(index="merged_topic_label", columns="emotion", values="proportion").fillna(0)
    pivoted.to_csv(os.path.join(app_folder, "emotion_distribution_by_topic.csv"))

    print(f"Finished processing {app_id}")

data_dir = "data"
app_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for app_id in app_ids:
    app_folder = os.path.join(data_dir, app_id)
    if is_app_preprocessed(app_id, data_dir=data_dir):
        print(f"✅ Skipping {app_id} — already processed.")
        continue

    review_csv = os.path.join(app_folder, "reviews.csv")
    if os.path.exists(review_csv):
        process_reviews_file(review_csv, app_folder, app_id)
    else:
        print(f"⚠️ No reviews.csv found for {app_id}")
