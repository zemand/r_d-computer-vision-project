"""Завантажує ParcelBar і InventBar (~510 МБ) з Kaggle у data/raw/parcelbar/ і data/raw/inventbar/.

Джерело: https://www.kaggle.com/datasets/chommaphatma/inventbar-and-parcelbar-benchmarkbarcodedatasets
Ліцензія: CC BY-SA 4.0 (за метаданими Kaggle).
Просять цитувати: T. Kamnardsiri, P. Charoenkwan, C. Malang, R. Wudhikarn,
"1D Barcode Detection: Novel Benchmark Datasets and Comprehensive Comparison of
Deep Convolutional Neural Network Approaches", Sensors 22(22), 8788, 2022.

Запуск з кореня проєкту:  python scripts/download_kaggle.py
"""

import shutil
from pathlib import Path

import kagglehub

DATASET = "chommaphatma/inventbar-and-parcelbar-benchmarkbarcodedatasets"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"


def main():
    # kagglehub качає в свій кеш і повертає шлях; публічний набір не потребує акаунта
    source = Path(kagglehub.dataset_download(DATASET))
    for name in ["ParcelBar", "InventBar"]:
        target = RAW / name.lower()
        if target.exists():
            print("Вже є:", target)
            continue
        shutil.copytree(source / name, target)
        print("Скопійовано:", target, len(list(target.glob("*.jpg"))), "фото")


if __name__ == "__main__":
    main()
