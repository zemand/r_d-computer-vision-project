"""Стенд пошкоджень: 8 типів × 5 рівнів сили (за зразком ImageNet-C).

Кожна функція: (сіре зображення uint8, рівень 1..5, seed) -> пошкоджене зображення.
Рівень 1 = ледь помітно, 5 = дуже сильно. Параметри рівнів у таблицях нижче:
їх можна підкрутити, якщо рівні вийдуть надто легкими чи важкими.
"""

import cv2
import numpy as np

LEVELS = [1, 2, 3, 4, 5]


def _to_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def dark(image, level, seed=0):
    """Темрява: менше світла + більше шуму сенсора (як у камери при слабкому світлі)."""
    brightness = [0.6, 0.4, 0.25, 0.15, 0.08][level - 1]
    noise_sigma = [2, 3, 4, 5, 6][level - 1]
    rng = np.random.default_rng(seed)
    result = image.astype(np.float32) * brightness
    result = result + rng.normal(0, noise_sigma, image.shape)
    return _to_uint8(result)


def uneven(image, level, seed=0):
    """Нерівне світло: один край кадру темніший (градієнт поперек смуг)."""
    darkest = [0.7, 0.5, 0.35, 0.2, 0.1][level - 1]
    width = image.shape[1]
    gradient = np.linspace(1.0, darkest, width).astype(np.float32)
    result = image.astype(np.float32) * gradient[np.newaxis, :]
    return _to_uint8(result)


def glare(image, level, seed=0):
    """Відблиск: яскрава пляма (як від лампи на глянці), що пересвічує частину коду."""
    strength = [80, 130, 180, 230, 255][level - 1]
    radius_part = [0.10, 0.15, 0.20, 0.25, 0.30][level - 1]
    rng = np.random.default_rng(seed)
    height, width = image.shape
    # центр плями випадково біля центру коду
    center_x = width * rng.uniform(0.35, 0.65)
    center_y = height * rng.uniform(0.35, 0.65)
    radius = min(height, width) * radius_part
    ys, xs = np.mgrid[0:height, 0:width]
    distance2 = (xs - center_x) ** 2 + (ys - center_y) ** 2
    spot = strength * np.exp(-distance2 / (2 * radius ** 2))
    return _to_uint8(image.astype(np.float32) + spot)


def noise(image, level, seed=0):
    """Шум сенсора (Gaussian)."""
    sigma = [5, 10, 20, 30, 45][level - 1]
    rng = np.random.default_rng(seed)
    return _to_uint8(image.astype(np.float32) + rng.normal(0, sigma, image.shape))


def motion_blur(image, level, seed=0):
    """Розмиття від руху (горизонтально, поперек смуг: найгірший випадок для 1D)."""
    length = [3, 5, 9, 13, 17][level - 1]
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel)


def defocus(image, level, seed=0):
    """Розфокус: камера не навелась."""
    sigma = [1.0, 1.5, 2.0, 3.0, 4.0][level - 1]
    return cv2.GaussianBlur(image, (0, 0), sigma)


def jpeg(image, level, seed=0):
    """Артефакти JPEG-стиснення (лекція 5)."""
    quality = [50, 30, 20, 10, 5][level - 1]
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def perspective(image, level, seed=0):
    """Нахил камери: прямокутник коду стає трапецією (лекція 7)."""
    shift_part = [0.05, 0.10, 0.15, 0.20, 0.25][level - 1]
    height, width = image.shape
    shift = width * shift_part
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # праву сторону "відсуваємо" від камери: вона стає коротшою
    target = np.float32([[0, 0], [width, shift], [width, height - shift], [0, height]])
    matrix = cv2.getPerspectiveTransform(source, target)
    # краї заповнюємо сусідніми пікселями: біле 255 виглядало б як відблиск
    return cv2.warpPerspective(image, matrix, (width, height),
                               borderMode=cv2.BORDER_REPLICATE)


DEGRADATIONS = {
    "dark": dark,
    "uneven": uneven,
    "glare": glare,
    "noise": noise,
    "motion_blur": motion_blur,
    "defocus": defocus,
    "jpeg": jpeg,
    "perspective": perspective,
}
