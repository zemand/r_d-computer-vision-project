"""Пайплайни: класичний (завжди однаковий), адаптивний (за діагнозом), каскад."""

from src import enhance
from src.diagnose import diagnose

# Параметри класичного пайплайну. Ноутбук 05 підбирає кращі через Optuna
DEFAULT_PARAMS = {
    "flat_kernel_part": 0.08,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "unsharp_amount": 1.0,
}


def classic_pipeline(image, params=None):
    """Завжди ті самі кроки: відблиск -> освітлення -> контраст -> різкість."""
    if params is None:
        params = DEFAULT_PARAMS
    result = enhance.glare_inpaint(image)
    result = enhance.flat_field(result, kernel_part=params["flat_kernel_part"])
    result = enhance.clahe(result, clip=params["clahe_clip"], tile=params["clahe_tile"])
    result = enhance.unsharp(result, amount=params["unsharp_amount"])
    return result


def adaptive_pipeline(image, thresholds=None):
    """Спершу діагноз, потім лише ті кроки, які потрібні цьому кадру."""
    problems = diagnose(image, thresholds)
    result = image
    if "noisy" in problems:
        # 3×3, а не 5×5: більше ядро "з'їдає" тонкі смуги (перевірено)
        result = enhance.median(result, size=3)
    # великий відблиск = локально надто яскравий фон, тому теж flat-field.
    # inpaint замальовував пляму разом зі смугами (перевірено), його лишаємо
    # окремим методом для порівняння в ноутбуці 03
    if "glare" in problems or "uneven" in problems:
        result = enhance.flat_field(result)
    if "dark" in problems:
        result = enhance.gamma(result)
    if "low_contrast" in problems:
        result = enhance.clahe(result)
    if "blurry" in problems:
        result = enhance.unsharp(result)
    return result


# Порядок спроб у каскаді: від найдешевшого до найдорожчого
CASCADE_STEPS = ["raw", "adaptive", "gamma", "clahe", "flat_field",
                 "glare_inpaint", "unsharp", "classic", "sauvola"]


def apply_step(image, step):
    """Застосувати один крок за назвою."""
    if step == "raw":
        return image
    if step == "adaptive":
        return adaptive_pipeline(image)
    if step == "classic":
        return classic_pipeline(image)
    return enhance.ENHANCERS[step](image)


def cascade_decode(image, decoder):
    """Пробуємо кроки по черзі, поки декодер щось не прочитає.

    Повертає (результати декодера, назва кроку, що спрацював, кількість спроб).
    Увага: "прочитав" не завжди означає "прочитав правильно", це перевіряє evaluate.
    """
    attempts = 0
    for step in CASCADE_STEPS:
        attempts = attempts + 1
        results = decoder(apply_step(image, step))
        if len(results) > 0:
            return results, step, attempts
    return [], None, attempts
