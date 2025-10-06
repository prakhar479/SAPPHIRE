import os
import pandas as pd
import h5py
from tqdm import tqdm
import warnings

# --- CONFIGURATION ---
# ⚠️ IMPORTANT: Update these paths to match your system
MSD_ROOT_PATH = "./"
OUTPUT_PARQUET_PATH = "msd_step1_full_features.parquet"
# ---------------------

# Suppress pandas warning
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_all_h5_files(basedir):
    """Recursively find all HDF5 files."""
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(basedir)
        for f in files
        if f.endswith(".h5")
    ]


def extract_full_data_from_h5(file_path):
    """Extracts a comprehensive set of features from a single HDF5 file."""
    try:
        with h5py.File(file_path, "r") as h5f:
            # ... (all the helper functions and extraction logic from the previous answer)
            # Helper function to safely extract scalar values
            def get_scalar(path):
                return path[0] if len(path) > 0 else None

            # Helper function to safely extract and decode string values
            def get_string(path):
                return path[0].decode("UTF-8") if len(path) > 0 else None

            # Helper function to extract array values
            def get_array(path):
                return path[:].tolist() if len(path) > 0 else None

            # --- Analysis Group ---
            analysis = h5f["analysis"]
            song_analysis = analysis["songs"]

            # --- Metadata Group ---
            metadata = h5f["metadata"]
            song_metadata = metadata["songs"]

            # --- MusicBrainz Group ---
            musicbrainz = h5f["musicbrainz"]
            song_musicbrainz = musicbrainz["songs"]

            data = {
                "track_id": get_string(song_analysis["track_id"]),
                "artist_name": get_string(song_metadata["artist_name"]),
                "title": get_string(song_metadata["title"]),
                "release": get_string(song_metadata["release"]),
                "song_id": get_string(song_metadata["song_id"]),
                "audio_md5": get_string(analysis["songs"]["audio_md5"]),
                # 'artist_mbid': get_string(song_musicbrainz['artist_mbid']),
                "artist_playmeid": get_scalar(song_metadata["artist_playmeid"]),
                "artist_terms": get_array(metadata["artist_terms"]),
                "artist_terms_freq": get_array(metadata["artist_terms_freq"]),
                "artist_terms_weight": get_array(metadata["artist_terms_weight"]),
                "similar_artists": get_array(metadata["similar_artists"]),
                "duration": get_scalar(song_analysis["duration"]),
                "song_hotttnesss": get_scalar(song_metadata["song_hotttnesss"]),
                "year": get_scalar(song_musicbrainz["year"]),
                "danceability": get_scalar(song_analysis["danceability"]),
                "energy": get_scalar(song_analysis["energy"]),
                "end_of_fade_in": get_scalar(song_analysis["end_of_fade_in"]),
                "start_of_fade_out": get_scalar(song_analysis["start_of_fade_out"]),
                "loudness": get_scalar(song_analysis["loudness"]),
                "key": get_scalar(song_analysis["key"]),
                "key_confidence": get_scalar(song_analysis["key_confidence"]),
                "mode": get_scalar(song_analysis["mode"]),
                "mode_confidence": get_scalar(song_analysis["mode_confidence"]),
                "tempo": get_scalar(song_analysis["tempo"]),
                "time_signature": get_scalar(song_analysis["time_signature"]),
                "time_signature_confidence": get_scalar(
                    song_analysis["time_signature_confidence"]
                ),
                "bars_start": get_array(analysis["bars_start"]),
                "bars_confidence": get_array(analysis["bars_confidence"]),
                "beats_start": get_array(analysis["beats_start"]),
                "beats_confidence": get_array(analysis["beats_confidence"]),
                "sections_start": get_array(analysis["sections_start"]),
                "sections_confidence": get_array(analysis["sections_confidence"]),
                "tatums_start": get_array(analysis["tatums_start"]),
                "tatums_confidence": get_array(analysis["tatums_confidence"]),
                "segments_start": get_array(analysis["segments_start"]),
                "segments_confidence": get_array(analysis["segments_confidence"]),
                "segments_pitches": get_array(analysis["segments_pitches"]),
                "segments_timbre": get_array(analysis["segments_timbre"]),
                "segments_loudness_max": get_array(analysis["segments_loudness_max"]),
                "segments_loudness_max_time": get_array(
                    analysis["segments_loudness_max_time"]
                ),
                "segments_loudness_start": get_array(
                    analysis["segments_loudness_start"]
                ),
                "track_7digitalid": get_array(song_metadata["track_7digitalid"]),
            }
            return data
    except Exception as e:
        # **MODIFICATION**: Print the error for the specific file that fails
        print(f"Could not process file {os.path.basename(file_path)}. Error: {e}")
        return None


# --- MAIN EXECUTION ---
print("🚀 Starting Step 1: Extracting FULL features from MSD...")

all_files = get_all_h5_files(MSD_ROOT_PATH)
if not all_files:
    print(f"❌ ERROR: No .h5 files found in '{MSD_ROOT_PATH}'. Please check your path.")
else:
    print(f"Found {len(all_files)} HDF5 files to process.")

    all_song_data = []
    for f in tqdm(all_files, desc="Processing HDF5 Files"):
        data = extract_full_data_from_h5(f)
        if data:
            all_song_data.append(data)

    # **MODIFICATION**: Check if any data was successfully extracted before proceeding
    if not all_song_data:
        print(
            "\n❌ ERROR: Failed to extract data from any of the HDF5 files. The output file was not created."
        )
        print("Please check the error messages above to diagnose the issue.")
    else:
        # Convert to a DataFrame and save
        df = pd.DataFrame(all_song_data)
        # This line is now safe because we know the DataFrame is not empty
        df.dropna(subset=["track_id"], inplace=True)
        df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
        df.to_csv("CHECK.csv", index=False)

        print(
            f"\n✅ Step 1 Complete! Saved {len(df)} tracks to '{OUTPUT_PARQUET_PATH}'"
        )
        print("DataFrame Info:")
        df.info()
