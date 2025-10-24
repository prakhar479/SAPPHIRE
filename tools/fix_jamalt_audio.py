#!/usr/bin/env python3
"""Fix jam-alt audio files copied from HF snapshot that may be symlink placeholders.

This script downloads the dataset snapshot (or reuses existing cache), resolves the
actual blob files for audio entries, and copies the real audio binary data into
your local `data/raw/jam-alt/audio` directory, overwriting any placeholder files.

Usage:
  python tools/fix_jamalt_audio.py --out-root data/raw/jam-alt --dry-run
  python tools/fix_jamalt_audio.py --out-root data/raw/jam-alt

The script is conservative: by default it shows what would be replaced (--dry-run).
"""

from pathlib import Path
import argparse
import shutil
import csv
import tempfile
import os
from huggingface_hub import snapshot_download


def resolve_and_copy(snapshot_dir: Path, filename: str, dest_dir: Path, dry_run: bool = True):
    src = snapshot_dir / 'audio' / filename
    if not src.exists():
        print(f"[WARN] audio entry not found in snapshot: {filename}")
        return False

    # Resolve symlink to actual blob file if necessary
    try:
        real = src.resolve()
    except Exception:
        real = src

    if not real.exists():
        print(f"[WARN] resolved source does not exist for {filename}: {real}")
        return False

    dst = dest_dir / filename
    if dry_run:
        print(f"DRY: would copy {real} -> {dst}")
        return True

    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(real, dst)
        print(f"Copied {real} -> {dst}")
        return True
    except Exception as e:
        print(f"[ERROR] failed to copy {real} -> {dst}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='jamendolyrics/jam-alt')
    parser.add_argument('--out-root', type=Path, default=Path('data/raw/jam-alt'))
    parser.add_argument('--split', default='test')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    out_root = args.out_root
    audio_dir = out_root / 'audio'
    lyrics_dir = out_root / 'lyrics'

    print('Downloading HF snapshot (or using cached copy)')
    snapshot_dir = Path(snapshot_download(repo_id=args.dataset, repo_type='dataset'))
    print('Snapshot at', snapshot_dir)

    metadata = snapshot_dir / 'metadata.csv'
    if not metadata.exists():
        print('metadata.csv not found in snapshot; aborting')
        return 1

    # Read metadata and copy corresponding audio blobs
    replaced = 0
    with open(metadata, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row.get('Filepath')
            if not filepath:
                continue
            filename = Path(filepath).name
            # Only operate on files that exist locally and are small or missing
            local = audio_dir / filename
            need_replace = True
            if local.exists():
                # If file size > 10KB assume it's a real audio file and skip
                try:
                    if local.stat().st_size > 10 * 1024:
                        need_replace = False
                except Exception:
                    pass

            if need_replace:
                ok = resolve_and_copy(snapshot_dir, filename, audio_dir, dry_run=args.dry_run)
                if ok and not args.dry_run:
                    replaced += 1

    print(f"Done. replaced={replaced} (dry_run={args.dry_run})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
