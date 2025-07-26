import os
import pandas as pd
from google_play_scraper import reviews, Sort
from tqdm import tqdm
import time

# Configuration
NUM_REVIEWS = 5000
LANG = "en"
COUNTRY = "us"
BASE_DIR = "data"  # Folder to store all apps

# Load your top apps list
apps_df = pd.read_csv("data/top_apps.csv")  # Must have 'app_id' column

# Ensure base data directory exists
os.makedirs(BASE_DIR, exist_ok=True)

for _, row in tqdm(apps_df.iterrows(), total=len(apps_df), desc="Scraping apps"):
    app_id = row["app_id"]
    app_folder = os.path.join(BASE_DIR, app_id)
    os.makedirs(app_folder, exist_ok=True)

    out_path = os.path.join(app_folder, "reviews.csv")
    if os.path.exists(out_path):
        print(f"✔️  Skipping {app_id} (already exists)")
        continue

    try:
        result, _ = reviews(
            app_id,
            lang=LANG,
            country=COUNTRY,
            sort=Sort.NEWEST,
            count=NUM_REVIEWS,
            filter_score_with=None
        )
        df = pd.DataFrame(result)
        df = df[["reviewId", "userName", "content", "score", "thumbsUpCount", "reviewCreatedVersion", "at"]]
        df.columns = ["id", "user", "text", "rating", "likes", "app_version", "date"]
        df.to_csv(out_path, index=False)
        print(f"✅ Saved {len(df)} reviews for {app_id}")
        
        time.sleep(2)
    except Exception as e:
        print(f"❌ Error scraping {app_id}: {e}")
