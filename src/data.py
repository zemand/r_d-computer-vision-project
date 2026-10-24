"""Дані: синтетичні баркоди з відомим вмістом, твої фото, Arte-Lab."""

import itertools
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import zxingcpp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Формати, з якими працюємо (назви однакові в усьому проєкті)
FORMATS = ["EAN13", "CODE128", "QR"]

# Відповідність наших назв форматам zxing-cpp
ZXING_FORMATS = {
    "EAN13": zxingcpp.BarcodeFormat.EAN13,
    "CODE128": zxingcpp.BarcodeFormat.Code128,
    "QR": zxingcpp.BarcodeFormat.QRCode,
}


def make_text(code_format, rng):
    """Випадковий вміст коду потрібного формату."""
    if code_format == "EAN13":
        # 482 = префікс GS1 України; не починаємо з 0, щоб zxing не сприйняв як UPC-A.
        # 12 цифр: 13-ту (контрольну) zxing-cpp додає сам
        return "482" + "".join(rng.integers(0, 10, size=9).astype(str))
    if code_format == "CODE128":
        # схоже на номер відправлення: UA + 10 цифр
        return "UA" + "".join(rng.integers(0, 10, size=10).astype(str))
    if code_format == "QR":
        number = rng.integers(100000, 999999)
        return "https://example.com/item/" + str(number)
    raise ValueError("Невідомий формат: " + code_format)


def barcode_to_image(barcode, module_px=3, border=40):
    """Малює код zxing-cpp як ідеальне зображення (сірий uint8, біле поле навколо)."""
    image = np.array(barcode.to_image(scale=module_px))
    return cv2.copyMakeBorder(image, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=255)


def render_code(text, code_format, module_px=3, border=40):
    """Ідеальне зображення коду за його текстом."""
    barcode = zxingcpp.create_barcode(text, ZXING_FORMATS[code_format])
    return barcode_to_image(barcode, module_px, border)


def camera_look(image, seed=0):
    """Робить ідеальний код схожим на кадр камери: реальні рівні, розмиття, шум.

    - Папір на фото не буває ідеально білим (255), а фарба ідеально чорною (0):
      беремо приблизно 230 і 30. Інакше білий фон виглядає як "пересвіт".
    - Без легкого розмиття детектор OpenCV не бачить ідеально різкі коди
      (перевірено на початку проєкту), і baseline виходить хибно поганим.
    """
    rng = np.random.default_rng(seed)
    levels = 30 + image.astype(np.float32) * (200.0 / 255.0)
    blurred = cv2.GaussianBlur(levels, (3, 3), 0)
    noise = rng.normal(0, 2, image.shape)
    result = blurred.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def generate_synthetic(n_per_format=20, seed=42):
    """Список зразків: по n кодів кожного формату, з відомим вмістом.

    Кожен зразок це словник: id, format, text, image.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for code_format, index in itertools.product(FORMATS, range(n_per_format)):
        barcode = zxingcpp.create_barcode(make_text(code_format, rng),
                                          ZXING_FORMATS[code_format])
        samples.append({
            "id": code_format + "_" + str(index).zfill(3),
            "format": code_format,
            "text": barcode.text,      # повний текст (для EAN-13 з контрольною цифрою)
            "image": camera_look(barcode_to_image(barcode), seed=seed + index),
        })
    return samples


def resize_max_side(image, max_side=1280):
    """Зменшує велике фото з телефона, зберігаючи пропорції."""
    height, width = image.shape[:2]
    biggest = max(height, width)
    if biggest <= max_side:
        return image
    scale = max_side / biggest
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def load_own_photos(folder=None):
    """Твої фото з data/raw/own/ + labels.csv.

    labels.csv: product,code,format,package (один рядок на товар).
    Файли: <product>_<light>.jpg, наприклад can_01_dark.jpg.
    """
    if folder is None:
        folder = DATA_DIR / "raw" / "own"
    folder = Path(folder)
    labels_file = folder / "labels.csv"
    if not labels_file.exists():
        print("Немає", labels_file, "- пропускаю власні фото")
        return []

    # dtype=str: щоб pandas не перетворив код 4820... на число і не загубив нулі
    labels = pd.read_csv(labels_file, dtype=str).set_index("product")

    samples = []
    for path in sorted(folder.glob("*.jpg")):
        # can_01_dark -> product = can_01, light = dark
        name_parts = path.stem.rsplit("_", 1)
        if len(name_parts) != 2 or name_parts[0] not in labels.index:
            print("Пропускаю (немає в labels.csv):", path.name)
            continue
        product, light = name_parts
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Не вдалося прочитати:", path.name)
            continue
        label = labels.loc[product]
        samples.append({
            "id": path.stem,
            "format": label["format"].upper(),
            "text": label["code"].strip(),
            "light": light,
            "package": label["package"],
            "image": resize_max_side(image),
        })
    return samples


def load_artelab(folder=None, limit=None):
    """Фото EAN-13 з Arte-Lab (без маски). Вміст коду невідомий: text = None."""
    if folder is None:
        folder = DATA_DIR / "raw" / "artelab"
    folder = Path(folder)
    if not folder.exists():
        print("Немає", folder, "- запусти scripts/download_artelab.py")
        return []

    samples = []
    paths = sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.png"))
    for path in paths:
        # маски детекції лежать поруч з фото, їх пропускаємо
        if "mask" in str(path).lower() or "detection" in path.parent.name.lower():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        samples.append({
            "id": path.stem,
            "format": "EAN13",
            "text": None,
            "image": resize_max_side(image),
        })
        if limit is not None and len(samples) >= limit:
            break
    return samples


def label_by_consensus(samples):
    """Для фото без відомого вмісту (Arte-Lab): правильна відповідь = те, що
    OpenCV і zxing-cpp ОДНОСТАЙНО прочитали на оригіналі.

    Фото, які прочитав лише один декодер або жоден, відкидаємо.
    Обмеження: набір зсувається в бік "легких" фото (на Arte-Lab лишається ~260 з 365).
    """
    from src.decoders import decode_opencv, decode_zxing

    labeled = []
    for sample in samples:
        opencv_texts = [r["text"] for r in decode_opencv(sample["image"]) if r["format"] == "EAN13"]
        zxing_texts = [r["text"] for r in decode_zxing(sample["image"]) if r["format"] == "EAN13"]
        if len(opencv_texts) == 1 and opencv_texts == zxing_texts:
            new_sample = dict(sample)
            new_sample["text"] = opencv_texts[0]
            labeled.append(new_sample)
    return labeled
