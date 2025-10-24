import os
from huggingface_hub import snapshot_download

# 1. Define the repository ID and the local folder
dataset_repo_id = "jamendolyrics/jam-alt"
local_dir = "jamendo_download"

print(f"Starting download of '{dataset_repo_id}' to '{local_dir}'...")
print("This script will resolve pointers and download the full audio files.")

# 2. Download the data
#
# --- THIS IS THE CORRECTED PART ---
# The files are nested in language-specific subfolders (es, en, fr, de)
#
snapshot_download(
    repo_id=dataset_repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["subsets/*/audio/*.mp3", "subsets/*/parquet/*.parquet"],
)

print("\n--- Download complete! ---")
print(f"All files are in the '{local_dir}' folder, organized in 'subsets'.")
