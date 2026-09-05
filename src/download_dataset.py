"""
download_dataset.py - Dataset Download Helper & Setup Guide.
Part of the GNN-BERT Music Context Understanding project (CSE425).

Supported Datasets (PDF Section 3 & Table 1):
1. GTZAN (Primary Audio Dataset - Easy Baseline)
   - 1,000 audio tracks (30 seconds each), 10 genres.
   - Size: ~1.2 GB
   - Download Link: http://marsyas.info/index.html / Kaggle / HuggingFace
2. MagnaTagATune (Multi-label Tags Dataset)
   - 25,877 clips with 188 tags.
   - Download: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
3. MusicCaps (Text Captions & Cross-Modal Alignment)
   - 5,521 clips with expert text descriptions by Google.
   - Download: https://huggingface.co/datasets/google/MusicCaps
"""

import os
import sys
import argparse
import urllib.request
import tarfile
import zipfile


GTZAN_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
MUSICCAPS_CSV_URL = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"


def download_file_with_progress(url: str, dest_path: str):
    """Download a remote file with progress printing and standard headers."""
    print(f"[*] Downloading from fast mirror: {url}")
    print(f"[*] Destination: {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", -1))
        downloaded = 0
        block_size = 1024 * 1024 # 1 MB chunks

        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = int(downloaded * 100 / total_size)
                sys.stdout.write(f"\r  Progress: {percent:3d}% [{downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB]")
            else:
                sys.stdout.write(f"\r  Downloaded: {downloaded / (1024*1024):.1f} MB")
            sys.stdout.flush()

    print("\n[OK] Download complete!")


def setup_gtzan(data_dir: str = "data/raw/gtzan"):
    """
    Download and extract the GTZAN Genre Collection.
    """
    archive_path = os.path.join(data_dir, "genres.tar.gz")
    if not os.path.exists(archive_path):
        print("\n=== DOWNLOADING GTZAN GENRE DATASET (~1.2 GB) ===")
        print("Tip: If download is slow on university Wi-Fi, you can also download via Kaggle:")
        print("  kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification")
        try:
            download_file_with_progress(GTZAN_URL, archive_path)
            print("[*] Extracting genres.tar.gz...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=data_dir)
            print(f"[OK] GTZAN successfully extracted to {data_dir}/genres/")
        except Exception as e:
            print(f"[!] Direct download encountered an issue: {e}")
            print("[*] Manual alternative: Place audio WAV/MP3 files into data/raw/genres/<genre_name>/*.wav")
    else:
        print(f"[OK] GTZAN archive already exists at {archive_path}")


def setup_musiccaps(data_dir: str = "data/raw/musiccaps"):
    """
    Download the MusicCaps metadata annotations (Google's expert music captions).
    """
    csv_path = os.path.join(data_dir, "musiccaps.csv")
    if not os.path.exists(csv_path):
        print("\n=== DOWNLOADING MUSICCAPS METADATA (Captions & Tags) ===")
        try:
            download_file_with_progress(MUSICCAPS_CSV_URL, csv_path)
            print(f"[OK] MusicCaps metadata saved to {csv_path}")
        except Exception as e:
            print(f"[!] Error downloading MusicCaps: {e}")
    else:
        print(f"[OK] MusicCaps CSV already exists at {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and configure music datasets.")
    parser.add_argument("--dataset", type=str, default="all", choices=["gtzan", "musiccaps", "all"], help="Dataset to download")
    args = parser.parse_args()

    print("=== Music Dataset Downloader & Setup ===")
    if args.dataset in ["gtzan", "all"]:
        setup_gtzan()
    if args.dataset in ["musiccaps", "all"]:
        setup_musiccaps()

    print("\nDataset configuration instructions ready!")
