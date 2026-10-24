"""Завантажує датасет Arte-Lab "1D barcode extended" (~225 МБ) у data/raw/artelab/.

Джерело: https://artelab.dista.uninsubria.it/downloads/datasets/barcode/hough_barcode_1d/hough_barcode_1d.html
Ліцензія: CC BY 3.0 (за списком github.com/BenSouchet/barcode-datasets).
Просять цитувати: A. Zamberletti, I. Gallo, S. Albertini,
"Robust Angle Invariant 1D Barcode Detection", ACPR 2013.

Запуск з кореня проєкту:  python scripts/download_artelab.py
"""

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URL = ("https://artelab.dista.uninsubria.it/downloads/datasets/barcode/"
       "hough_barcode_1d/1d_barcode_extended_plain.zip")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "artelab"
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "artelab.zip"


def main():
    if TARGET_DIR.exists() and any(TARGET_DIR.iterdir()):
        print("Вже є:", TARGET_DIR)
        return

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print("Завантажую", URL)
        response = requests.get(URL, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        with open(ZIP_PATH, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))

    print("Розпаковую в", TARGET_DIR)
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(TARGET_DIR)
    ZIP_PATH.unlink()
    print("Готово")


if __name__ == "__main__":
    main()
