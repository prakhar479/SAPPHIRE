#!/usr/bin/env python3
"""Download jam-alt audio files from the original URLs listed in metadata.csv.

Reads the HF dataset snapshot's `metadata.csv`, extracts the URL and Filepath
columns, and downloads audio files into <out_root>/audio/<Filepath> when the
local file is missing or smaller than a threshold.

Usage:
  python tools/download_jamalt_audio_from_urls.py --out-root data/raw/jam-alt --dry-run
  python tools/download_jamalt_audio_from_urls.py --out-root data/raw/jam-alt --workers 8
"""

from pathlib import Path
import argparse
import csv
import time
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except Exception:
    requests = None

from huggingface_hub import snapshot_download


def download_url_to(path_url: str, dst: Path, timeout: int = 30, retries: int = 3) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 10 * 1024:
        return True

    for attempt in range(1, retries + 1):
        try:
            if requests is None:
                # fallback to urllib
                from urllib.request import urlopen
                with urlopen(path_url, timeout=timeout) as r, open(dst, 'wb') as f:
                    shutil.copyfileobj(r, f)
                return True

            headers = {'User-Agent': 'sapphire-downloader/1.0 (+https://github.com)'}
            with requests.get(path_url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                with open(dst, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            last_exc = e
            time.sleep(1 * attempt)
    print(f"Failed to download {path_url}: {last_exc}")
    return False


def worker(entry, snapshot_dir: Path, out_audio: Path, dry_run: bool):
    url = entry.get('URL')
    filepath = entry.get('Filepath')
    if not url or not filepath:
        return False, filepath
    dst = out_audio / Path(filepath).name
    if dry_run:
        need = not (dst.exists() and dst.stat().st_size > 10 * 1024)
        return need, filepath

    ok = download_url_to(url, dst)
    return ok, filepath if ok else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='jamendolyrics/jam-alt')
    parser.add_argument('--out-root', type=Path, default=Path('data/raw/jam-alt'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    out_root = args.out_root
    out_audio = out_root / 'audio'
    out_audio.mkdir(parents=True, exist_ok=True)

    print('Fetching HF snapshot (cached)...')
    snapshot_dir = Path(snapshot_download(repo_id=args.dataset, repo_type='dataset'))
    metadata = snapshot_dir / 'metadata.csv'
    if not metadata.exists():
        print('metadata.csv not found in snapshot; aborting')
        return 1

    entries = []
    with open(metadata, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)

    if args.limit > 0:
        entries = entries[args.start:args.start + args.limit]
    else:
        entries = entries[args.start:]

    print(f'Will process {len(entries)} entries (dry_run={args.dry_run})')

    results = []
    if args.dry_run:
        for e in entries[:50]:
            need, filepath = worker(e, snapshot_dir, out_audio, dry_run=True)
            if need:
                print('DRY -> need download:', filepath)
        print('Dry run complete (showing first 50).')
        return 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, e, snapshot_dir, out_audio, False): e for e in entries}
        for fut in as_completed(futures):
            ok, filepath = fut.result()
            if ok:
                results.append(filepath)

    print(f'Downloaded {len(results)} files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
