"""Швидка перевірка, що весь проєкт працює (~1 хв).

Запуск з кореня проєкту:  python scripts/smoke_test.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import generate_synthetic, load_artelab, load_own_photos  # noqa: E402
from src.decoders import DECODERS, decoders_for  # noqa: E402
from src.diagnose import diagnose  # noqa: E402
from src.evaluate import run_benchmark, summarize  # noqa: E402

errors = 0

# 1. Синтетика: кожен чистий код має прочитатися хоч одним декодером
samples = generate_synthetic(n_per_format=2, seed=1)
for sample in samples:
    read_ok = False
    for name in decoders_for(sample["format"]):
        for result in DECODERS[name](sample["image"]):
            if result["text"] == sample["text"]:
                read_ok = True
    if not read_ok:
        print("ПОМИЛКА: не прочитано", sample["id"])
        errors = errors + 1
    if diagnose(sample["image"]) != []:
        print("УВАГА: чистий кадр має діагноз", sample["id"], diagnose(sample["image"]))
print("Синтетика:", len(samples), "зразків")

# 2. Маленький прогін сітки
table = run_benchmark(samples, ["raw", "adaptive", "cascade"], levels=[1, 5])
print(summarize(table, by=["method"]).to_string(index=False))

# 3. Реальні дані (якщо є)
print("Arte-Lab:", len(load_artelab(limit=5)), "фото (перші 5)")
print("Власні фото:", len(load_own_photos()))

if errors > 0:
    print("FAILED:", errors)
    sys.exit(1)
print("OK")
