import streamlit as st
import pandas as pd
import os
import altair as alt

st.set_page_config(page_title="App Review Insights in Top 50 Apps", layout="wide")


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

st.title(f"📱 Feedback Explorer for: {selected_app_name}")

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
    st.markdown(f"**🧵 Number of topics identified:** {len(topics_df)}\n")
    col1, col2 = st.columns([2, 1]) 

    with col1:
        sort_col = st.selectbox("Sort topics by:", topics_df.columns.tolist(), index=0)
    with col2:
        ascending = st.radio("Sort order:", ["Descending", "Ascending"], horizontal=True) == "Ascending"

    sorted_topics_df = topics_df.sort_values(by=sort_col, ascending=ascending)

    st.dataframe(sorted_topics_df, use_container_width=True)


    if "review_count" in topics_df.columns:
        top_topic = topics_df.sort_values("review_count", ascending=False).iloc[0]
        st.info(f"🔍 Most mentioned topic: `{top_topic['merged_topic_label']}` with `{top_topic['review_count']}` reviews")


# --- Pain Points Tab ---
with tabs[1]:
    st.subheader("Topics with High Negative Emotion")
    pain_points = pain_summary_df[pain_summary_df["pain_point_flag"] == True]
    st.write("These topics had >40% of reviews with anger, sadness, fear, or disgust.")
    st.markdown(f"**🚨 Number of flagged pain point topics:** {len(pain_points)}")
    if not pain_points.empty:
        top_pain = pain_points["merged_topic_label"].value_counts().idxmax()
        pain_reviews = reviews_df[reviews_df["merged_topic_label"].isin(pain_points["merged_topic_label"])]
        common_emotion = pain_reviews["emotion"].value_counts().idxmax()
        st.info(f"Top pain point topic: `{top_pain}` — Most frequent emotion: `{common_emotion}`")

    st.dataframe(pain_points, use_container_width=True)

    st.subheader("Emotion Breakdown by Pain Point Topic")
    pain_reviews = pain_reviews.dropna(subset=["emotion", "merged_topic_label"])
    grouped = pain_reviews.groupby(["merged_topic_label", "emotion"]).size().reset_index(name="count")

    stacked_bar = alt.Chart(grouped).mark_bar().encode(
        x=alt.X("merged_topic_label:N", title="Topic", sort="-y"),
        y=alt.Y("count:Q", stack="normalize", title="Proportion"),
        color=alt.Color("emotion:N")
    ).properties(width=800, height=400)

    st.altair_chart(stacked_bar, use_container_width=True)

# --- Emotion Distribution Tab ---
with tabs[2]:
    st.subheader("Emotion Distribution by Topic")
    melted = emotion_dist_df.melt(id_vars="merged_topic_label", var_name="emotion", value_name="proportion")
    st.markdown(f"**📊 Total topics:** {emotion_dist_df.shape[0]}")
    dominant_emotion = melted.groupby("emotion")["proportion"].mean().idxmax()
    st.info(f"Most dominant emotion across topics: `{dominant_emotion}`")

    chart = alt.Chart(melted).mark_bar().encode(
        x=alt.X("merged_topic_label:N", title="Topic Label", sort="-y"),
        y=alt.Y("proportion:Q", stack="normalize"),
        color=alt.Color("emotion:N")
    ).properties(width=800, height=400)

    st.altair_chart(chart, use_container_width=True)

    st.subheader("Emotion Trends Over Time")
    time_series = reviews_df.dropna(subset=["emotion", "date"]).copy()
    time_series["date"] = pd.to_datetime(time_series["date"])
    time_series_grouped = time_series.groupby([pd.Grouper(key="date", freq="W"), "emotion"]).size().reset_index(name="count")

    top_emotion_time = time_series_grouped.groupby("emotion")["count"].sum().idxmax()
    st.markdown(f"**📈 Emotion with most mentions over time:** `{top_emotion_time}`")

    emotion_line = alt.Chart(time_series_grouped).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("count:Q", title="Number of Reviews"),
        color=alt.Color("emotion:N")
    ).properties(width=800, height=400)

    st.altair_chart(emotion_line, use_container_width=True)

    st.subheader("Topic-to-Emotion Heatmap")
    melted = emotion_dist_df.melt(
    id_vars="merged_topic_label",
    var_name="emotion",
    value_name="proportion")

    # Drop bad rows
    melted = melted.dropna(subset=["proportion"])
    melted = melted[melted["proportion"] > 0]

    heatmap = alt.Chart(melted).mark_rect().encode(
    x=alt.X("emotion:N", title="Emotion"),
    y=alt.Y("merged_topic_label:N", title="Topic"),
    color=alt.Color("proportion:Q", scale=alt.Scale(scheme="reds")),
    tooltip=["merged_topic_label", "emotion", "proportion"]
    ).properties(width=800, height=400)

    st.altair_chart(heatmap, use_container_width=True)


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

    st.markdown(f"**🔎 Showing {len(filtered)} reviews after filters**")
    st.dataframe(filtered[["date", "text", "emotion", "vader_sentiment", "merged_topic_label", "pain_point"]], use_container_width=True)

    st.subheader("VADER Sentiment Histogram")
    valid_sent = reviews_df.dropna(subset=["vader_sentiment"])
    pos_ratio = (valid_sent["vader_sentiment"] > 0.05).mean()
    neg_ratio = (valid_sent["vader_sentiment"] < -0.05).mean()
    st.markdown(f"**🙂 Positive reviews:** `{pos_ratio:.1%}` &nbsp;&nbsp;&nbsp; **☹️ Negative reviews:** `{neg_ratio:.1%}`")

    vader_hist = alt.Chart(valid_sent).mark_bar().encode(
        x=alt.X("vader_sentiment:Q", bin=alt.Bin(maxbins=30), title="VADER Sentiment Score"),
        y=alt.Y("count():Q", title="Number of Reviews"),
        tooltip=["count()"]
    ).properties(width=800, height=400)

    st.altair_chart(vader_hist, use_container_width=True)
