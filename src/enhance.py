"""Класичні методи покращення кадру (лекції 1-4, 8, 9).

Кожна функція: сіре зображення uint8 -> сіре зображення uint8.
Параметри мають значення за замовчуванням; Optuna (ноутбук 05) підбирає кращі.
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def _odd(value, minimum=3):
    """Найближче непарне число не менше minimum (розміри ядер мають бути непарні)."""
    value = max(int(value), minimum)
    if value % 2 == 0:
        value = value + 1
    return value


def gamma(image, value=None):
    """Gamma-корекція (лекція 2). value=None: підібрати так, щоб середнє стало ~128.

    Лише освітлює (value <= 1): світлий кадр не затемнюємо.
    """
    if value is None:
        mean = min(max(image.mean() / 255.0, 0.01), 0.99)
        value = np.log(0.5) / np.log(mean)
        value = float(np.clip(value, 0.2, 1.0))
    table = (np.linspace(0, 1, 256) ** value) * 255
    return cv2.LUT(image, table.astype(np.uint8))


def clahe(image, clip=2.0, tile=8):
    """CLAHE: вирівнювання гістограми по шматочках (лекція 2)."""
    tool = cv2.createCLAHE(clipLimit=clip, tileGridSize=(int(tile), int(tile)))
    return tool.apply(image)


def background_map(image, kernel_part=0.08):
    """Карта освітлення фону: фільтр максимуму прибирає темні смуги, лишається папір.

    Розмір ядра має бути більшим за найширшу смугу коду.
    """
    size = _odd(min(image.shape) * kernel_part, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    background = cv2.dilate(image, kernel)
    return cv2.GaussianBlur(background, (size, size), 0)


def flat_field(image, kernel_part=0.08):
    """Flat-field: ділимо кадр на карту освітлення фону (лекція 3).

    Там, де світла мало, ділимо на мале число: ділянка стає світлішою.
    """
    background = background_map(image, kernel_part).astype(np.float32) + 1.0
    result = image.astype(np.float32) / background * 230.0
    return np.clip(result, 0, 255).astype(np.uint8)


def tophat(image, kernel_part=0.06):
    """Black-hat: виділяє темні смуги на світлому фоні незалежно від освітлення."""
    size = _odd(min(image.shape) * kernel_part, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    # інвертуємо: смуги знову темні, фон світлий
    return cv2.normalize(255 - blackhat, None, 0, 255, cv2.NORM_MINMAX)


def otsu(image):
    """Глобальний поріг Otsu. Антиприклад: при нерівному світлі гірше, ніж нічого."""
    threshold, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_threshold(image, block_part=0.08, c=10):
    """Адаптивний поріг: свій поріг для кожної ділянки (лекція 9: integral image)."""
    block = _odd(min(image.shape) * block_part, 11)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, c)


def sauvola(image, window_part=0.08, k=0.2):
    """Sauvola: адаптивний поріг з урахуванням локального контрасту."""
    window = _odd(min(image.shape) * window_part, 11)
    threshold = threshold_sauvola(image, window_size=window, k=k)
    return ((image > threshold) * 255).astype(np.uint8)


def glare_inpaint(image, level=250, radius=5):
    """Відблиск: маска пересвічених пікселів -> домальовуємо їх із сусідів.

    TODO: на фото з білим папером, що теж дає 255, маска може захопити фон.
    """
    mask = (image >= level).astype(np.uint8) * 255
    if mask.sum() == 0:
        return image
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def unsharp(image, amount=1.0, sigma=2.0):
    """Unsharp mask: підсилює різкість (лекція 3)."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def median(image, size=3):
    """Медіанний фільтр: прибирає шум, зберігаючи краї (лекція 3)."""
    return cv2.medianBlur(image, _odd(size))


# Методи "по одному": для порівняння кожного з кожним пошкодженням
ENHANCERS = {
    "gamma": gamma,
    "clahe": clahe,
    "flat_field": flat_field,
    "tophat": tophat,
    "otsu": otsu,
    "adaptive_threshold": adaptive_threshold,
    "sauvola": sauvola,
    "glare_inpaint": glare_inpaint,
    "unsharp": unsharp,
    "median": median,
}
