import pandas as pd
import sqlite3
from tqdm import tqdm

# --- CONFIGURATION ---
# ⚠️ IMPORTANT: Update these paths
# This should be the output from your first feature extraction step
INPUT_PARQUET_PATH = 'msd_step1_full_features.parquet' 
MUSIXMATCH_DB_PATH = './mxm_dataset.db'
OUTPUT_PARQUET_PATH = 'msd_with_musixmatch_lyrics.parquet'
# ---------------------

print("🚀 Starting: Augmenting with MusixMatch bag-of-words lyrics...")

# Load the data from the previous step
df = pd.read_parquet(INPUT_PARQUET_PATH)

# Connect to the SQLite database
conn = sqlite3.connect(MUSIXMATCH_DB_PATH)

# Query the lyrics data
# The 'lyrics' table has (track_id, word, count)
# We group by track_id to get all words for each song into a single string
query = "SELECT track_id, group_concat(word || ':' || count) FROM lyrics GROUP BY track_id"
lyrics_df = pd.read_sql_query(query, conn)
conn.close()

# Rename the column for clarity
lyrics_df = lyrics_df.rename(columns={'group_concat(word || \':\' || count)': 'lyrics_bow'})

print(f"Found bag-of-words lyrics for {len(lyrics_df)} tracks in the MusixMatch database.")

# Merge the lyrics data with our main DataFrame
df_merged = pd.merge(df, lyrics_df, on='track_id', how='inner')

# Save the augmented DataFrame
df_merged.to_parquet(OUTPUT_PARQUET_PATH, index=False)

print(f"\n✅ Complete! Merged lyrics for {len(df_merged)} tracks.")
print(f"Saved to '{OUTPUT_PARQUET_PATH}'")

print("\n--- Example Output ---")
# Display the track info and the new 'lyrics_bow' column
print(df_merged[['track_id', 'title', 'artist_name', 'lyrics_bow']].head())