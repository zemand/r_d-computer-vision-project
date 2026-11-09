"""Дані: синтетичні баркоди з відомим вмістом, пари для навчання, Arte-Lab."""

import itertools
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

from src.degradations import DEGRADATIONS, LEVELS

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


def camera_look(image, seed=0):
    """Робить ідеальний код схожим на кадр камери: реальні рівні, легке розмиття, шум.

    Папір на фото не ідеально білий (255), фарба не ідеально чорна (0): беремо ~230 і ~30.
    Без легкого розмиття детектор OpenCV не бачить ідеально різкі коди.
    """
    rng = np.random.default_rng(seed)
    levels = 30 + image.astype(np.float32) * (200.0 / 255.0)
    blurred = cv2.GaussianBlur(levels, (3, 3), 0)
    result = blurred + rng.normal(0, 2, image.shape)
    return np.clip(result, 0, 255).astype(np.uint8)


def generate_synthetic(n_per_format=20, seed=42, module_sizes=(2, 3, 4)):
    """Список зразків: по n кодів кожного формату, з відомим вмістом.

    Кожен зразок це словник: id, format, text, image.
    module_sizes: товщина найтоншої смуги в пікселях (різний масштаб коду в кадрі).
    """
    rng = np.random.default_rng(seed)
    samples = []
    for code_format, index in itertools.product(FORMATS, range(n_per_format)):
        barcode = zxingcpp.create_barcode(make_text(code_format, rng),
                                          ZXING_FORMATS[code_format])
        module_px = int(rng.choice(module_sizes))
        samples.append({
            "id": code_format + "_" + str(index).zfill(3),
            "format": code_format,
            "text": barcode.text,      # повний текст (для EAN-13 з контрольною цифрою)
            "image": camera_look(barcode_to_image(barcode, module_px), seed=seed + index),
        })
    return samples


def make_training_batch(samples, batch_size, rng, crop=128):
    """Пачка пар для навчання: (зіпсований шматок, чистий шматок), обидва float32 0..1.

    Для кожної пари: випадковий код -> випадкове пошкодження і рівень (або без пошкодження,
    щоб мережа вчилась не псувати хороші кадри) -> однаковий випадковий шматок crop×crop
    з обох картинок -> випадковий поворот на 90° (смуги бувають і вертикальні, і горизонтальні).
    Повертає масиви форми (batch, 1, crop, crop), як чекає PyTorch.
    """
    names = list(DEGRADATIONS) + ["clean"]
    inputs = np.zeros((batch_size, 1, crop, crop), np.float32)
    targets = np.zeros((batch_size, 1, crop, crop), np.float32)
    for i in range(batch_size):
        clean = samples[rng.integers(len(samples))]["image"]
        name = names[rng.integers(len(names))]
        if name == "clean":
            damaged = clean
        else:
            level = int(rng.choice(LEVELS))
            damaged = DEGRADATIONS[name](clean, level, seed=int(rng.integers(1_000_000)))

        top = rng.integers(clean.shape[0] - crop + 1)
        left = rng.integers(clean.shape[1] - crop + 1)
        damaged = damaged[top:top + crop, left:left + crop]
        clean = clean[top:top + crop, left:left + crop]

        turns = int(rng.integers(4))
        inputs[i, 0] = np.rot90(damaged, turns) / 255.0
        targets[i, 0] = np.rot90(clean, turns) / 255.0
    return inputs, targets


def resize_max_side(image, max_side=1280):
    """Зменшує велике фото, зберігаючи пропорції."""
    height, width = image.shape[:2]
    biggest = max(height, width)
    if biggest <= max_side:
        return image
    scale = max_side / biggest
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def load_artelab(folder=None, limit=None):
    """Фото EAN-13 з Arte-Lab (без масок). Вміст коду невідомий: text = None."""
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
    Обмеження: набір зсувається в бік "легких" фото.
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


def split_by_code(samples, test_part=0.3, seed=0):
    """Ділить фото на train і test так, щоб один товар (код) був лише в одній частині.

    В Arte-Lab той самий товар знято багато разів (254 фото, 47 кодів). Якщо ділити
    за файлами, мережа бачила б у train майже ті самі кадри, що в test, і результат
    був би завищений.
    """
    codes = sorted(set(sample["text"] for sample in samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(codes)

    test_codes = set()
    test_count = 0
    for code in codes:
        if test_count >= test_part * len(samples):
            break
        test_codes.add(code)
        test_count += sum(sample["text"] == code for sample in samples)

    train = [sample for sample in samples if sample["text"] not in test_codes]
    test = [sample for sample in samples if sample["text"] in test_codes]
    return train, test
