import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
from tqdm import tqdm
import textstat
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Global models
analyzer = SentimentIntensityAnalyzer()
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,
    model_kwargs={"torch_dtype": torch.float32}
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()

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
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error labeling topic {topic_num}: {e}")
        return "Unknown"

def process_reviews_file(app_id: str, base_dir="data"):
    input_path = os.path.join(base_dir, app_id, "reviews.csv")
    output_path = os.path.join(base_dir, app_id, f"{app_id}_reviews_preprocessed.csv")
    app_folder = os.path.join(base_dir, app_id)
    os.makedirs(app_folder, exist_ok=True)

    if not os.path.exists(input_path):
        print(f"❌ Missing input file for {app_id}")
        return

    print(f"🔄 Processing {app_id}...")
    df = pd.read_csv(input_path)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().astype(bool)]

    # Basic preprocessing
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["text_length"] = df["text"].str.len()
    df["vader_sentiment"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["emotion"] = batch_emotion_labels(df["text"].tolist())
    df["readability"] = df["text"].apply(textstat.flesch_reading_ease)

    # Keyword flags
    bug_keywords = ['bug', 'crash', 'glitch', 'error', 'issue', 'broken', 'freeze']
    request_keywords = ['add', 'feature', 'please', 'would like', 'suggest', 'can you', 'request']
    df['contains_bug_keywords'] = df['text'].str.lower().apply(lambda x: any(k in x for k in bug_keywords))
    df['contains_request_keywords'] = df['text'].str.lower().apply(lambda x: any(k in x for k in request_keywords))

    # Topic modeling
    print("🔍 Running BERTopic...")
    topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
    topics, probs = topic_model.fit_transform(df['text'].tolist())
    df['topic'] = topics
    df['topic_prob'] = probs

    # GPT topic labeling
    docs = df['text'].tolist()
    topic_info = topic_model.get_topic_info()
    labels = {}

    for topic_num in tqdm(topic_info['Topic']):
        if topic_num == -1:
            continue
        top_words = [word for word, _ in topic_model.get_topic(topic_num)[:10]]
        sample_reviews = [docs[i] for i, t in enumerate(topics) if t == topic_num][:3]
        if len(sample_reviews) < 3:
            continue
        label = suggest_label(topic_num, top_words, sample_reviews)
        labels[topic_num] = label

    # Merge topics with same label
    label_to_new_topic = {}
    new_topic_id = 0
    topic_remap = {}

    for old_topic, label in labels.items():
        norm_label = label.strip().lower()
        if norm_label not in label_to_new_topic:
            label_to_new_topic[norm_label] = new_topic_id
            new_topic_id += 1
        topic_remap[old_topic] = label_to_new_topic[norm_label]

    df["merged_topic"] = df["topic"].map(topic_remap)
    df["merged_topic_label"] = df["merged_topic"].map(
        lambda x: [k for k, v in label_to_new_topic.items() if v == x][0] if x in df["merged_topic"].unique() else "Unknown"
    )

    # Pain point detection
    negative_emotions = ['anger', 'sadness', 'fear', 'disgust']
    topic_counts = df.groupby('merged_topic_label').size()
    neg_counts = df[df['emotion'].isin(negative_emotions)].groupby('merged_topic_label').size()
    neg_proportion = (neg_counts / topic_counts).fillna(0)
    threshold = 0.4
    pain_points = neg_proportion[neg_proportion > threshold]
    df['pain_point'] = df['merged_topic_label'].isin(pain_points.index)

    # Save processed reviews
    df.to_csv(output_path, index=False)

    # Save topic summary with pain point flag
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

    # Save pain summary
    pain_summary = pd.DataFrame({
        'negative_proportion': neg_proportion,
        'total_reviews': topic_counts,
        'pain_point_flag': neg_proportion > threshold
    }).reset_index()
    pain_summary.to_csv(os.path.join(app_folder, "pain_point_summary.csv"), index=False)

    # Save normalized emotion distribution
    emotion_dist = (
        df[["merged_topic_label", "emotion"]]
        .value_counts(normalize=True)
        .reset_index(name="proportion")
    )
    pivoted = emotion_dist.pivot(index="merged_topic_label", columns="emotion", values="proportion").fillna(0)
    pivoted.to_csv(os.path.join(app_folder, "emotion_distribution_by_topic.csv"))
