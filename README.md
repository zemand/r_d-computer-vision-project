# Robustness in Industrial Computer Vision

Адаптивна обробка зображень для стійкого розпізнавання баркодів/QR
в умовах нестабільного промислового освітлення.

## Задача

На лініях сортування і складах баркоди часто не читаються через темряву, нерівне
світло, відблиски на глянцевій упаковці, розмиття і шум. Проєкт вимірює, як ці
пошкодження впливають на decode rate (частку правильно прочитаних кодів), і
порівнює способи це виправити: класичну обробку кадру, адаптивну обробку
(спершу діагноз кадру, потім лише потрібні кроки) і каскад спроб.

## Підхід (пайплайн)

```
кадр -> діагностика (яскравість, контраст, пересвіт, нерівність, різкість, шум)
     -> вибір обробки (gamma, flat-field, CLAHE, median, unsharp ...)
     -> декодер (OpenCV / WeChat QR / zxing-cpp) -> рядок
```

Стенд пошкоджень (за зразком ImageNet-C): 8 типів × 5 рівнів сили:
`dark`, `uneven`, `glare`, `noise`, `motion_blur`, `defocus`, `jpeg`, `perspective`.

## Структура

```
src/
  data.py          синтетичні коди з відомим вмістом, власні фото, Arte-Lab
  decoders.py      OpenCV barcode (EAN/UPC), WeChat QR, zxing-cpp -> один формат
  degradations.py  стенд пошкоджень
  enhance.py       класичні методи: gamma, CLAHE, flat-field, top-hat, Otsu,
                   adaptive threshold, Sauvola, glare inpaint, unsharp, median
  diagnose.py      метрики кадру і діагноз
  pipeline.py      classic / adaptive / cascade
  evaluate.py      прогін сітки -> таблиця, підсумки, графіки
scripts/
  download_artelab.py   завантаження Arte-Lab (~225 МБ) у data/raw/artelab/
  smoke_test.py         швидка перевірка, що все працює
notebooks/
  01_baseline              декодери на чистих кодах
  02_degradations          галерея пошкоджень, baseline-сітка
  03_classic_enhancement   кожен метод проти кожного пошкодження
  04_adaptive              діагностика, adaptive, cascade
  05_tuning_optuna         підбір параметрів (train/test)
  06_real_photos           Arte-Lab і власні фото
  07_results               графіки й таблиці для презентації
results/            CSV і PNG з прогонів
```

## Дані

| Джерело | Що | Ліцензія |
|---|---|---|
| Синтетика (`src/data.py`) | EAN-13 (префікс 482), Code128, QR, генеруються zxing-cpp | власні |
| [Arte-Lab 1D barcode extended](https://artelab.dista.uninsubria.it/downloads/datasets/barcode/hough_barcode_1d/hough_barcode_1d.html) | 365 фото EAN-13 | CC BY 3.0; цитувати: Zamberletti, Gallo, Albertini, *Robust Angle Invariant 1D Barcode Detection*, ACPR 2013 |
| Власні фото | баркоди продуктів при різному світлі | власні |

Зображення в git не зберігаються (`data/raw/` у `.gitignore`).

### Власні фото

`data/raw/own/<product>_<light>.jpg`, де `light` = `normal`, `dark`, `glare`, `side`, `bright`.
Правильні відповіді в `data/raw/own/labels.csv`, один рядок на товар:

```
product,code,format,package
can_01,4820000000000,EAN13,can
```

## Як запустити

```bash
source ~/venvs/cv/bin/activate
pip install -r requirements.txt
python scripts/download_artelab.py     # один раз
python scripts/smoke_test.py           # ~1 хв, має закінчитися "OK"
jupyter lab notebooks/                 # ноутбуки по порядку 01 -> 07
```

## Результати

TODO: після фінальних прогонів (рисунки з `notebooks/07_results.ipynb`).

## Висновки та обмеження

TODO. Вже відомо:
- `cv2.barcode` читає лише EAN/UPC; Code128 читає тільки zxing-cpp.
- Детектор OpenCV не бачить ідеально різкі синтетичні коди, тому синтетика проходить
  через `camera_look` (реальні рівні, легке розмиття, шум).
- Variance of Laplacian завищується шумом, тому шум діагностується окремо.
- Arte-Lab не містить вмісту кодів: правильна відповідь = консенсус двох декодерів
  на оригіналі, що зсуває набір у бік легких фото.
- Камера телефона сама коригує експозицію (HDR, нічний режим), промислова камера ні.
