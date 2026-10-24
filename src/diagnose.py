"""Діагностика кадру: що з ним не так (темно, відблиск, розмито...)."""

import cv2
import numpy as np

from src.enhance import background_map

# Пороги за замовчуванням. TODO: відкалібрувати в ноутбуці 04 на наших даних
THRESHOLDS = {
    "dark_brightness": 90,       # середня яскравість нижче -> темно
    "low_contrast": 80,          # різниця перцентилів 95-5 нижче -> низький контраст
    "glare_saturated": 0.005,    # частка пікселів >= 250 вище -> відблиск
    "uneven": 0.25,              # нерівність освітлення вище -> нерівне світло
    "blurry_sharpness": 300,     # variance of Laplacian нижче -> розмито
    "noisy": 16,                 # рівень шуму вище -> шумно (слабкий шум декодери
                                 # переносять краще без фільтра, перевірено)
}


def measure(image):
    """Числові метрики якості кадру."""
    brightness = float(image.mean())
    p5, p95 = np.percentile(image, [5, 95])
    contrast = float(p95 - p5)
    # пересвіт рахуємо після медіанного фільтра: окремі шумові пікселі не відблиск
    smooth = cv2.medianBlur(image, 5)
    saturated = float((smooth >= 250).mean())
    # різкість: variance of Laplacian (Pech-Pacheco et al., 2000)
    # обмеження: шум теж дає великі значення, тому шум оцінюємо окремо
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    # шум: різниця з медіанним фільтром лише на рівних ділянках (без країв смуг)
    difference = np.abs(image.astype(np.float32) - smooth)
    edges = cv2.morphologyEx(smooth, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    flat = edges < 20
    if flat.any():
        noise_level = float(np.median(difference[flat]))
    else:
        noise_level = 0.0
    # нерівність: наскільки різниться освітлення ФОНУ в частинах кадру 4×4
    # (смуги коду спершу прибираємо, інакше темний код = "нерівне світло")
    background = background_map(image)
    small = cv2.resize(background, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    unevenness = float((small.max() - small.min()) / (small.max() + 1.0))
    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturated": saturated,
        "sharpness": sharpness,
        "unevenness": unevenness,
        "noise": noise_level,
    }


def diagnose(image, thresholds=None):
    """Список проблем кадру, наприклад ['dark', 'blurry']. Порожній = все добре."""
    if thresholds is None:
        thresholds = THRESHOLDS
    metrics = measure(image)
    problems = []
    if metrics["saturated"] > thresholds["glare_saturated"]:
        problems.append("glare")
    if metrics["unevenness"] > thresholds["uneven"]:
        problems.append("uneven")
    if metrics["brightness"] < thresholds["dark_brightness"]:
        problems.append("dark")
    if metrics["contrast"] < thresholds["low_contrast"]:
        problems.append("low_contrast")
    if metrics["noise"] > thresholds["noisy"]:
        problems.append("noisy")
    # розмитість перевіряємо лише без сильного шуму: шум "обманює" метрику різкості
    elif metrics["sharpness"] < thresholds["blurry_sharpness"]:
        problems.append("blurry")
    return problems
