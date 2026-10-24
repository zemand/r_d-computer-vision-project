"""Дані: синтетичні баркоди з відомим вмістом, твої фото, Arte-Lab."""

import csv
from pathlib import Path

import cv2
import numpy as np
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


def ean13_checksum(digits12):
    """Контрольна цифра EAN-13 для рядка з 12 цифр."""
    total = 0
    for position in range(12):
        digit = int(digits12[position])
        # непарні позиції (1, 3, 5...) з вагою 1, парні з вагою 3
        if position % 2 == 0:
            total = total + digit
        else:
            total = total + digit * 3
    return str((10 - total % 10) % 10)


def make_text(code_format, rng):
    """Випадковий вміст коду потрібного формату."""
    if code_format == "EAN13":
        # 482 = префікс GS1 України; не починаємо з 0, щоб zxing не сприйняв як UPC-A
        digits = "482"
        for _ in range(9):
            digits = digits + str(rng.integers(0, 10))
        return digits + ean13_checksum(digits)
    if code_format == "CODE128":
        # схоже на номер відправлення: UA + 10 цифр
        text = "UA"
        for _ in range(10):
            text = text + str(rng.integers(0, 10))
        return text
    if code_format == "QR":
        number = rng.integers(100000, 999999)
        return "https://example.com/item/" + str(number)
    raise ValueError("Невідомий формат: " + code_format)


def render_code(text, code_format, module_px=3, border=40):
    """Малює ідеальний код (сірий uint8, білий фон, поле навколо)."""
    barcode = zxingcpp.create_barcode(text, ZXING_FORMATS[code_format])
    image = np.array(barcode.to_image(scale=module_px))
    image = cv2.copyMakeBorder(image, border, border, border, border,
                               cv2.BORDER_CONSTANT, value=255)
    return image


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
    for code_format in FORMATS:
        for index in range(n_per_format):
            text = make_text(code_format, rng)
            image = render_code(text, code_format)
            image = camera_look(image, seed=seed + index)
            sample = {
                "id": code_format + "_" + str(index).zfill(3),
                "format": code_format,
                "text": text,
                "image": image,
            }
            samples.append(sample)
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

    labels = {}
    with open(labels_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[row["product"]] = row

    samples = []
    for path in sorted(folder.glob("*.jpg")):
        # can_01_dark -> product = can_01, light = dark
        name_parts = path.stem.rsplit("_", 1)
        if len(name_parts) != 2 or name_parts[0] not in labels:
            print("Пропускаю (немає в labels.csv):", path.name)
            continue
        product, light = name_parts
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Не вдалося прочитати:", path.name)
            continue
        sample = {
            "id": path.stem,
            "format": labels[product]["format"].upper(),
            "text": labels[product]["code"].strip(),
            "light": light,
            "package": labels[product]["package"],
            "image": resize_max_side(image),
        }
        samples.append(sample)
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
        opencv_texts = []
        for result in decode_opencv(sample["image"]):
            if result["format"] == "EAN13":
                opencv_texts.append(result["text"])
        zxing_texts = []
        for result in decode_zxing(sample["image"]):
            if result["format"] == "EAN13":
                zxing_texts.append(result["text"])
        if len(opencv_texts) == 1 and opencv_texts == zxing_texts:
            new_sample = dict(sample)
            new_sample["text"] = opencv_texts[0]
            labeled.append(new_sample)
    return labeled
