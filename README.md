# Robustness in Industrial Computer Vision

CNN-реставрація кадру перед декодером, щоб баркоди/QR читались при поганому світлі,
відблисках, шумі й розмитті.

## Задача

На лініях сортування і складах частина баркодів дає **No Read** через темряву, нерівне
світло, відблиски, шум і розмиття. Мета: підняти **decode rate** (частку правильно
прочитаних кодів), не змінюючи сам декодер.

## Підхід

```
кадр -> U-Net (реставрація) -> декодер (zxing-cpp / OpenCV / WeChat QR) -> рядок
```

- **U-Net** — маленький denoising convolutional encoder-decoder (~1.9 млн параметрів)
  зі skip connections, які зберігають тонкі смуги коду.
- **Навчання** на парах (зіпсований, чистий) із синтетичних кодів, реальних фото або їх суміші: 7 типів пошкоджень ×
  5 рівнів (`dark`, `uneven`, `glare`, `noise`, `motion_blur`, `defocus`, `jpeg`).
  ~10-12 хв на модель на Mac Mini M4 (MPS).
- **raw→cnn**: спершу читаємо сирий кадр, CNN запускаємо лише при No Read.

## Структура

```
src/
  data.py          синтетичні коди, пари для навчання, Arte-Lab
  degradations.py  7 типів пошкоджень × 5 рівнів
  decoders.py      zxing-cpp, OpenCV barcode, WeChat QR -> один формат
  model.py         U-Net і restore() для кадру будь-якого розміру
scripts/
  download_artelab.py   завантаження Arte-Lab у data/raw/artelab/
notebooks/
  01_train.ipynb        навчання трьох моделей -> models/unet_{synthetic,artelab,mix}.pt
  02_evaluate.ipynb     raw / cnn / raw→cnn для трьох моделей на Arte-Lab test і синтетиці
results/                CSV і графіки з 02_evaluate
```

## Дані

| Джерело | Що | Ліцензія |
|---|---|---|
| Синтетика (`src/data.py`) | EAN-13, Code128, QR з відомим вмістом; train seed 1, test seed 2 | власні |
| [Arte-Lab 1D barcode extended](https://artelab.dista.uninsubria.it/downloads/datasets/barcode/hough_barcode_1d/hough_barcode_1d.html) | фото EAN-13; поділ за товаром: train 167, test 87 | CC BY 3.0; Zamberletti, Gallo, Albertini, *Robust Angle Invariant 1D Barcode Detection*, ACPR 2013 |

Зображення і модель у git не зберігаються (`data/raw/`, `models/`).

## Як запустити

```bash
source ~/venvs/cv/bin/activate
pip install -r requirements.txt
python scripts/download_artelab.py     # один раз
jupyter lab notebooks/                 # 01_train -> 02_evaluate
```

## Результати

Три моделі з однаковою архітектурою: **syn** вчилась на синтетиці, **real** на 167 фото
Arte-Lab (28 товарів), **mix** на синтетиці й цих фото навпіл. Тест: 87 інших фото Arte-Lab
(19 товарів, жоден не з train), чисті + 7 пошкоджень × 5 рівнів.

| Метод | zxing-cpp | OpenCV | хибні декоди (OpenCV) |
|---|---|---|---|
| raw | 45% | 56% | 0.4% |
| raw→cnn_syn | 69% | 69% | 3.0% |
| raw→cnn_real | 66% | 65% | 0.8% |
| raw→cnn_mix | 68% | **70%** | **0.8%** |

![decode rate](results/fig_overall.png)
![приклади](results/examples.png)

- Приріст +20-25 п.п.; найбільше на темряві, нерівному світлі, шумі, JPEG.
- mix: decode rate як у syn, хибних декодів мало, як у real. syn "домальовує" текстури на фоні.
- Розмиття (defocus, motion_blur) на реальних фото не виправляє жодна модель.
- Вибір: raw→cnn_mix. Час CNN ~25 мс на фото 640×480 (Mac Mini M4, MPS), лише для кадрів з No Read.

Застосування: доповнення до промислового зчитувача, друга спроба для кадрів з No Read.

## Обмеження

- Мало реальних даних: 254 фото, 47 товарів; тест 87 фото, тому різниця в 2-3 п.п. ненадійна.
- Arte-Lab не містить вмісту кодів: правильна відповідь = консенсус OpenCV і zxing-cpp
  на оригіналі, тому в наборі лишаються переважно легкі фото.
- Пошкодження на Arte-Lab штучні (накладені на реальні фото).
- Нахил і перспективу CNN не виправляє.
