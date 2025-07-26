# 📱 Kid's App Feedback Explorer

This is a Streamlit web app for exploring user feedback on the top 50 kids apps from the Google Play Store.

## 🚀 Features

- Topic modeling of app reviews
- Emotion classification (anger, sadness, fear, etc.)
- Pain point detection based on emotion proportions
- Filtering and exploring raw reviews
- Interactive charts and summaries

## 🏗️ Folder Structure
.
├── app.py # Main Streamlit app
├── data/
│ ├── top_apps.csv # List of top apps with names and IDs
│ ├── <app_id>/
│ │ ├── <app_id>_reviews_preprocessed.csv
│ │ ├── topics.csv
│ │ ├── emotion_distribution_by_topic.csv
│ │ └── pain_point_summary.csv
├── requirements.txt # Dependencies
├── .gitignore
└── README.md
