# Music Feature Pipeline — README


## Quickstart


1. Create a Python 3.10 venv
python -m venv .venv
source .venv/bin/activate
2. pip install -r requirements.txt
(If essentia/madmom fail, install them on the host or remove them from requirements)
3. Optionally run `bash download_models.sh` to pre-cache SBERT model
4. Run a single file:
python music_multifeature_extract_v2.py --audio /path/to/song.wav --lyrics /path/to/lyrics.txt --out features.json
5. Batch process folder:
python batch_process.py /path/to/folder


## Notes
- CREPE improves pitch — enable by installing `crepe`.