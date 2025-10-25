import pandas as pd
import requests
from tqdm import tqdm
import os

# --- CONFIGURATION ---
# ⚠️ IMPORTANT: Update these paths
# Use the Parquet file that now contains 'track_7digitalid'
INPUT_PARQUET_PATH = "msd_step1_full_features.parquet"
AUDIO_OUTPUT_DIR = "./30sec_clips/"
# ---------------------

print("🚀 Starting: Downloading 30-second audio clips...")

# 1. Create the output directory if it doesn't exist
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
print(f"Audio clips will be saved in: {AUDIO_OUTPUT_DIR}")

# 2. Load your dataset
df = pd.read_parquet(INPUT_PARQUET_PATH)

# Filter out rows that are missing the 7digital ID
df_download = df[df["track_7digitalid"].notna() & (df["track_7digitalid"] > 0)].copy()
print(f"Found {len(df_download)} tracks with valid 7digital IDs to download.")


# 3. Loop through the DataFrame and download each clip
for index, row in tqdm(
    df_download.iterrows(), total=df_download.shape[0], desc="Downloading Clips"
):
    track_id = row["track_id"]
    seven_digital_id = int(row["track_7digitalid"])

    # Define the output path for the mp3 file
    output_path = os.path.join(AUDIO_OUTPUT_DIR, f"{track_id}.mp3")

    # Skip if the file already exists
    if os.path.exists(output_path):
        continue

    # Construct the preview URL
    # The format uses the 7digital ID to build the path
    url = f"https://previews.7digital.com/clips/34/{seven_digital_id}.clip.mp3"

    try:
        # Make the request to download the file
        response = requests.get(url, timeout=10)
        # Raise an exception if the request was not successful (e.g., 404 Not Found)
        response.raise_for_status()
        print(response)
        # Write the content to the file
        with open(output_path, "wb") as f:
            f.write(response.content)

    except requests.exceptions.RequestException as e:
        # This will catch network errors, timeouts, 404s, etc.
        # Silently continue, but you could print the error if you want:
        print(f"Could not download {track_id}. Error: {e}")
        pass

print("\n✅ Download process complete!")
