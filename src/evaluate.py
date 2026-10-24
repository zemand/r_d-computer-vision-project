"""Оцінка: проганяємо зразки через пошкодження, методи й декодери -> таблиця."""

import time
import zlib

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from src import enhance, pipeline
from src.decoders import DECODERS, decoders_for
from src.degradations import DEGRADATIONS, LEVELS


def check(results, true_text):
    """'correct' / 'wrong' (прочитав, але не те) / 'none' (нічого не прочитав)."""
    if len(results) == 0:
        return "none"
    for result in results:
        if result["text"] == true_text:
            return "correct"
    return "wrong"


def get_method(name):
    """Метод покращення за назвою: raw, classic, adaptive або один з enhance.ENHANCERS."""
    if name == "raw":
        return None
    if name == "classic":
        return pipeline.classic_pipeline
    if name == "adaptive":
        return pipeline.adaptive_pipeline
    return enhance.ENHANCERS[name]


def evaluate_image(image, true_text, code_format, method_name, decoder_names=None):
    """Один кадр, один метод, усі доречні декодери -> список рядків таблиці."""
    if decoder_names is None:
        decoder_names = decoders_for(code_format)
    rows = []
    for decoder_name in decoder_names:
        if decoder_name not in decoders_for(code_format):
            continue
        decoder = DECODERS[decoder_name]
        start = time.perf_counter()
        step = method_name
        if method_name == "cascade":
            results, step, attempts = pipeline.cascade_decode(image, decoder)
        else:
            method = get_method(method_name)
            if method is None:
                prepared = image
            else:
                prepared = method(image)
            results = decoder(prepared)
        ms = (time.perf_counter() - start) * 1000
        rows.append({
            "method": method_name,
            "decoder": decoder_name,
            "outcome": check(results, true_text),
            "ms": ms,
            "step": step,
        })
    return rows


def run_benchmark(samples, methods, degradations=None, levels=None,
                  decoder_names=None, include_clean=True):
    """Повна сітка: зразок × пошкодження × рівень × метод × декодер.

    Рівень 0 = чистий кадр без пошкодження.
    """
    if degradations is None:
        degradations = list(DEGRADATIONS)
    if levels is None:
        levels = LEVELS

    # спершу складаємо список усіх варіантів кадрів
    jobs = []
    for sample in samples:
        if include_clean:
            jobs.append((sample, "clean", 0))
        for degradation in degradations:
            for level in levels:
                jobs.append((sample, degradation, level))

    rows = []
    for sample, degradation, level in tqdm(jobs, desc="benchmark"):
        if degradation == "clean":
            image = sample["image"]
        else:
            # seed залежить від зразка й рівня: результат відтворюваний між запусками
            key = sample["id"] + "|" + degradation + "|" + str(level)
            seed = zlib.crc32(key.encode("utf-8"))
            image = DEGRADATIONS[degradation](sample["image"], level, seed=seed)
        for method_name in methods:
            image_rows = evaluate_image(image, sample["text"], sample["format"],
                                        method_name, decoder_names)
            for row in image_rows:
                row["sample"] = sample["id"]
                row["format"] = sample["format"]
                row["degradation"] = degradation
                row["level"] = level
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(table, by=None):
    """Частки correct / wrong / none і середній час, згруповані за колонками by."""
    if by is None:
        by = ["method", "decoder"]
    table = table.copy()
    table["correct"] = table["outcome"] == "correct"
    table["wrong"] = table["outcome"] == "wrong"
    summary = table.groupby(by).agg(
        decode_rate=("correct", "mean"),
        wrong_rate=("wrong", "mean"),
        ms=("ms", "mean"),
        n=("outcome", "size"),
    )
    return summary.reset_index()


def plot_curves(table, degradation, decoder, methods=None, ax=None):
    """Графік decode rate проти рівня пошкодження, лінія на кожен метод."""
    data = table[(table["degradation"].isin([degradation, "clean"]))
                 & (table["decoder"] == decoder)]
    if methods is not None:
        data = data[data["method"].isin(methods)]
    curves = summarize(data, by=["method", "level"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    for method_name in curves["method"].unique():
        line = curves[curves["method"] == method_name]
        ax.plot(line["level"], line["decode_rate"], marker="o", label=method_name)
    ax.set_title(degradation + " / " + decoder)
    ax.set_xlabel("рівень пошкодження (0 = чистий)")
    ax.set_ylabel("decode rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return ax
