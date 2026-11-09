"""Дані: синтетичні баркоди з відомим вмістом, пари для навчання, Arte-Lab."""

import itertools
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

from src.degradations import DEGRADATIONS, LEVELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Формати синтетики (назви однакові в усьому проєкті)
FORMATS = ["EAN13", "CODE128", "QR"]
# На реальних фото беремо ще й складські Code39 та короткі EAN-8
REAL_FORMATS = FORMATS + ["CODE39", "EAN8"]

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


def resize_max_side(image, max_side=1600):
    """Зменшує велике фото, зберігаючи пропорції."""
    height, width = image.shape[:2]
    biggest = max(height, width)
    if biggest <= max_side:
        return image
    scale = max_side / biggest
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


# Реальні фото: назва набору -> папка в data/raw/ і скрипт завантаження
PHOTO_SETS = {
    "artelab": "scripts/download_artelab.py",      # EAN-13, CC BY 3.0
    "parcelbar": "scripts/download_kaggle.py",     # Code128 на посилках, CC BY-SA 4.0
    "inventbar": "scripts/download_kaggle.py",     # товари на складі, CC BY-SA 4.0
}


def load_photos(name, limit=None):
    """Сірі фото набору з data/raw/<name>/. Вміст кодів невідомий: text = None."""
    folder = DATA_DIR / "raw" / name
    if not folder.exists():
        print("Немає", folder, "- запусти", PHOTO_SETS[name])
        return []

    samples = []
    paths = sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.png"))
    for path in paths:
        # маски детекції Arte-Lab лежать поруч з фото, їх пропускаємо
        if "mask" in str(path).lower() or "detection" in path.parent.name.lower():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        samples.append({
            "id": name + "_" + path.stem,
            "source": name,
            "format": None,
            "text": None,
            "image": resize_max_side(image),
        })
        if limit is not None and len(samples) >= limit:
            break
    return samples


def label_photos(samples):
    """Правильна відповідь для фото без відомого вмісту.

    Беремо фото, де zxing-cpp прочитав на оригіналі рівно ОДИН код з REAL_FORMATS.
    Для EAN-13 і EAN-8 вимагаємо ще й згоди OpenCV (консенсус двох декодерів).
    Code128 і QR читає лише zxing, але в них вбудована перевірка (контрольна сума / корекція
    помилок), тому хибне читання на чистому оригіналі малоймовірне. У Code39 контрольна
    сума необов'язкова, тож ризик трохи вищий.

    Фото без відповіді відкидаємо. Обмеження: набір зсувається в бік "легких" фото.
    Ці ж фото — хороші цілі для навчання: оригінал, який читається, достатньо чистий.
    """
    from src.decoders import decode_opencv, decode_zxing

    labeled = []
    for sample in samples:
        results = decode_zxing(sample["image"])
        if len(results) != 1 or results[0]["format"] not in REAL_FORMATS:
            continue
        text, code_format = results[0]["text"], results[0]["format"]
        if code_format in ["EAN13", "EAN8"]:
            opencv_texts = [r["text"] for r in decode_opencv(sample["image"])]
            if opencv_texts != [text]:
                continue
        new_sample = dict(sample)
        new_sample["text"] = text
        new_sample["format"] = code_format
        labeled.append(new_sample)
    return labeled


def load_real_split(test_part=0.3, seed=0):
    """Усі три набори: фото з відповіддю, поділені на train і test окремо в кожному наборі."""
    train, test = [], []
    for name in PHOTO_SETS:
        labeled = label_photos(load_photos(name))
        name_train, name_test = split_by_code(labeled, test_part, seed)
        print(f"{name}: {len(labeled)} фото з відповіддю -> train {len(name_train)}, test {len(name_test)}")
        train += name_train
        test += name_test
    return train, test


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
