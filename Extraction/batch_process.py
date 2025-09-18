#!/usr/bin/env python3
"""Batch processing utilities to convert a folder of songs into a Parquet/CSV table."""
import os, glob, json
import pandas as pd
from music_multifeature_extract_v2 import analyze




def process_directory(directory, out_csv='batch_features.parquet'):
    meta = []
    for audio in glob.iglob(os.path.join(directory, '**/*.*'), recursive=True):
        if audio.lower().endswith(('.mp3','.wav','.flac')):
            base, _ = os.path.splitext(audio)
            lyrics = base + '.txt'
            feats = analyze(audio_path=audio, lyrics_path=lyrics)
            # convert to flat table (example: take duration, tempo, mean MFCC 0)
            row = {'audio_path': audio}
            if 'acoustic' in feats:
                row['duration'] = feats['acoustic'].get('duration', None)
            row['tempo_bpm'] = feats.get('rhythm',{}).get('tempo_bpm', None)
            if 'lyrics' in feats:
                row['has_lyrics'] = True
            meta.append(row)
    df = pd.DataFrame(meta)
    df.to_parquet(out_csv)
    print('Saved', out_csv)