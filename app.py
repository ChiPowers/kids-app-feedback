import streamlit as st
import pandas as pd
import os
import altair as alt

st.set_page_config(page_title="Kid's App Feedback Explorer", layout="wide")

st.title("📱 Kid's App Feedback Explorer")

# --- Sidebar: App selection ---
data_dir = "data"
available_app_ids = [
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
]

# Load app names from top_apps.csv
top_apps_df = pd.read_csv("data/top_apps.csv")

# Filter to only apps with data folders
top_apps_df = top_apps_df[top_apps_df["app_id"].isin(available_app_ids)]

# Create name-to-ID map
app_name_to_id = dict(zip(top_apps_df["app_name"], top_apps_df["app_id"]))

# Sidebar dropdown showing app names
selected_app_name = st.sidebar.selectbox("Select an App", list(app_name_to_id.keys()))

# Get corresponding app_id
app_id = app_name_to_id[selected_app_name]

app_path = os.path.join(data_dir, app_id)
reviews_path = os.path.join(app_path, f"{app_id}_reviews_preprocessed.csv")
topics_path = os.path.join(app_path, "topics.csv")
emotion_dist_path = os.path.join(app_path, "emotion_distribution_by_topic.csv")
pain_summary_path = os.path.join(app_path, "pain_point_summary.csv")

@st.cache_data
def load_data(app_id):
    app_path = os.path.join(data_dir, app_id)
    reviews_path = os.path.join(app_path, f"{app_id}_reviews_preprocessed.csv")
    topics_path = os.path.join(app_path, "topics.csv")
    emotion_dist_path = os.path.join(app_path, "emotion_distribution_by_topic.csv")
    pain_summary_path = os.path.join(app_path, "pain_point_summary.csv")
    
    return (
        pd.read_csv(reviews_path),
        pd.read_csv(topics_path),
        pd.read_csv(emotion_dist_path),
        pd.read_csv(pain_summary_path)
    )

reviews_df, topics_df, emotion_dist_df, pain_summary_df = load_data(app_id)

# --- Tabs ---
tabs = st.tabs(["📊 Topic Summary", "😠 Pain Points", "🎭 Emotion Distribution", "📝 Raw Reviews"])

# --- Topic Summary Tab ---
with tabs[0]:
    st.subheader("Topic-Level Summary")
    st.dataframe(topics_df, use_container_width=True)

# --- Pain Points Tab ---
with tabs[1]:
    st.subheader("Topics with High Negative Emotion")
    pain_points = pain_summary_df[pain_summary_df["pain_point_flag"] == True]
    st.write("These topics had >40% of reviews with anger, sadness, fear, or disgust.")
    st.dataframe(pain_points, use_container_width=True)

# --- Emotion Distribution Tab ---
with tabs[2]:
    st.subheader("Emotion Distribution by Topic")
    melted = emotion_dist_df.melt(id_vars="merged_topic_label", var_name="emotion", value_name="proportion")

    chart = alt.Chart(melted).mark_bar().encode(
        x=alt.X("merged_topic_label:N", title="Topic Label", sort="-y"),
        y=alt.Y("proportion:Q", stack="normalize"),
        color=alt.Color("emotion:N")
    ).properties(width=800, height=400)

    st.altair_chart(chart, use_container_width=True)

# --- Raw Reviews Tab ---
with tabs[3]:
    st.subheader("All Reviews")
    col1, col2 = st.columns(2)

    topics = reviews_df["merged_topic_label"].dropna().unique().tolist()
    emotions = reviews_df["emotion"].dropna().unique().tolist()

    selected_topic = col1.selectbox("Filter by Topic", ["All"] + sorted(topics))
    selected_emotion = col2.selectbox("Filter by Emotion", ["All"] + sorted(emotions))

    filtered = reviews_df.copy()
    if selected_topic != "All":
        filtered = filtered[filtered["merged_topic_label"] == selected_topic]
    if selected_emotion != "All":
        filtered = filtered[filtered["emotion"] == selected_emotion]

    st.write(f"Showing {len(filtered)} reviews")
    st.dataframe(filtered[["date", "text", "emotion", "vader_sentiment", "merged_topic_label", "pain_point"]], use_container_width=True)