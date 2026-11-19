# -*- coding: utf-8 -*-
"""
================================================================================
 Лабораторная работа №4. Введение в нейронные сети
 Классификация рукописных цифр MNIST полносвязной сетью (Keras / TensorFlow)
================================================================================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ
--------------------
0. Повторяет пример из методички: один нейрон «вручную» на numpy
   (взвешенная сумма + сигмоида) для классификации пол по весу и росту.
1. Загружает MNIST (70 000 картинок 28x28), нормирует пиксели в [-1, 1],
   «вытягивает» картинку в вектор из 784 чисел, делает one-hot кодирование.
2. Базовая модель из методички: Dense(64, sigmoid) -> Dense(64, sigmoid)
   -> Dense(10, softmax), loss = categorical_crossentropy, 10 эпох.
   Обучаем с SGD и с Adam, показываем прогноз на 10 последних картинках,
   матрицу ошибок и примеры ошибок сети.
3. ЗАДАНИЕ 1: зависимость точности от числа скрытых слоёв (1..5) при
   фиксированных 64 нейронах в слое и остальных параметрах.
   Делаем для двух оптимизаторов (SGD и Adam) — так хорошо видна проблема
   затухающего градиента у глубоких сигмоидных сетей при обучении SGD.
4. ЗАДАНИЕ 2: зависимость точности от доли тестовой выборки 5%..50% (шаг 5%).
5. ЗАДАНИЕ 3: сравнение оптимизаторов SGD, SGD+momentum, RMSprop, Adam,
   Nadam, Adamax, Adagrad, Adadelta (кривые обучения + итоговая точность).

Каждый эксперимент повторяется для нескольких random seed (по умолчанию 3),
в таблицы пишется среднее и стандартное отклонение — так выводы не зависят
от случайной инициализации весов.

ЗАПУСК
------
    .venv\\Scripts\\python.exe lab4_solution.py           # полный прогон (~30-40 мин на CPU)
    .venv\\Scripts\\python.exe lab4_solution.py --quick   # быстрый прогон (1 seed, 3 эпохи)

Все результаты (CSV-таблицы и PNG-графики) сохраняются в папку results/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# Убираем служебные сообщения TensorFlow (оставляем только ошибки).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")           # строим графики без окон, сразу в PNG
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Пути и параметры
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Фиксированные («прочие») параметры — одинаковые во всех экспериментах,
# если явно не сказано иное. Именно так требует задание: меняем один фактор.
NEURONS = 64                # нейронов в каждом скрытом слое
ACTIVATION = "sigmoid"      # функция активации скрытых слоёв (как в методичке)
EPOCHS = 10                 # число эпох (как в методичке)
BATCH_SIZE = 32             # размер мини-батча (значение Keras по умолчанию)
TEST_SIZE = 0.25            # доля теста (как в методичке)
BASE_LAYERS = 2             # число скрытых слоёв в базовой модели
SEEDS = (42, 7, 2024)       # повторы для усреднения

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# 0. Один нейрон «вручную»
# ---------------------------------------------------------------------------
def single_neuron_demo() -> pd.DataFrame:
    """
    Нейрон = взвешенная сумма входов + смещение, пропущенная через функцию
    активации. Здесь веса подобраны вручную (как в методичке), обучения нет.
        r = w1*вес + w2*рост + b,   y = sigmoid(r) = 1 / (1 + e^(-r))
    Если y > 0.5 — класс «male», иначе «female».
    """
    data = {"Иван": [84, 180, 1], "Мария": [57, 165, 0], "Анна": [62, 170, 0]}
    w1, w2, b = 0.3, 0.1, -39

    rows = []
    for name, (weight, height, true_label) in data.items():
        r = w1 * weight + w2 * height + b          # взвешенная сумма
        y = 1 / (1 + np.exp(-r))                   # сигмоида
        rows.append({
            "Имя": name, "Вес": weight, "Рост": height,
            "r (сумма)": round(r, 2), "sigmoid(r)": round(float(y), 3),
            "Прогноз": "male" if y > 0.5 else "female",
            "Истина": "male" if true_label == 1 else "female",
        })
    df = pd.DataFrame(rows)
    print("\n[neuron] Один нейрон, веса заданы вручную:")
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "00_single_neuron.csv", index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------------------
# 1. Данные
# ---------------------------------------------------------------------------
def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает MNIST через sklearn.fetch_openml (как в методичке).
    Данные кэшируются в папку data/, поэтому скачиваются только один раз.
    X: (70000, 784) — яркость пикселей 0..255; y: (70000,) — цифры 0..9.
    """
    print("[data] Загружаю MNIST (первый раз скачивается ~15 МБ)...")
    mnist = fetch_openml("mnist_784", as_frame=False, data_home=str(DATA_DIR),
                         parser="auto")
    X = mnist.data.astype("float32")
    y = mnist.target.astype("int64")
    print(f"[data] X: {X.shape}, y: {y.shape}, пиксели {X.min():.0f}..{X.max():.0f}")
    return X, y


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Масштабирование в [-1, 1] по формуле из методички:
        X' = 2 * (X - min) / ptp - 1,  где ptp = max - min (peak to peak).
    Для пикселей min = 0, ptp = 255, т.е. X' = X / 127.5 - 1.
    Используем фиксированные границы 0..255, чтобы train и test
    масштабировались ОДИНАКОВО (в методичке min/ptp считаются отдельно
    по каждой выборке — для MNIST это даёт то же самое).
    """
    return 2.0 * (X - 0.0) / 255.0 - 1.0


def split(X, y, test_size: float, seed: int):
    """Разбиение на train/test + нормализация. stratify сохраняет доли классов."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    return normalize(X_train), normalize(X_test), y_train, y_test


# ---------------------------------------------------------------------------
# 2. Модель
# ---------------------------------------------------------------------------
def make_optimizer(name: str):
    """Создаёт оптимизатор Keras по короткому имени (каждый раз новый объект)."""
    name = name.lower()
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=0.01)
    if name == "sgd_momentum":
        return keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=0.001)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=0.001)
    if name == "nadam":
        return keras.optimizers.Nadam(learning_rate=0.001)
    if name == "adamax":
        return keras.optimizers.Adamax(learning_rate=0.001)
    if name == "adagrad":
        return keras.optimizers.Adagrad(learning_rate=0.001)
    if name == "adadelta":
        return keras.optimizers.Adadelta(learning_rate=0.001)
    raise ValueError(name)


def build_model(n_hidden: int = BASE_LAYERS, neurons: int = NEURONS,
                activation: str = ACTIVATION, optimizer: str = "adam") -> Sequential:
    """
    Полносвязная сеть (многослойный перцептрон):
        вход 784 -> [Dense(neurons, activation)] * n_hidden -> Dense(10, softmax)
    softmax на выходе превращает 10 чисел в вероятности классов (сумма = 1).
    """
    model = Sequential()
    model.add(Input(shape=(784,)))
    for _ in range(n_hidden):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=make_optimizer(optimizer),
                  metrics=["accuracy"])
    return model


def train_eval(X_train, y_train, X_test, y_test, *, n_hidden=BASE_LAYERS,
               optimizer="adam", seed=42, epochs=EPOCHS, track_test=False):
    """
    Обучает новую модель и возвращает (модель, словарь метрик, history).
    track_test=True — после каждой эпохи считаем качество на тесте, чтобы
    нарисовать кривые обучения. На подбор параметров это не влияет.
    """
    keras.utils.set_random_seed(seed)   # фиксирует random у python/numpy/tf
    model = build_model(n_hidden=n_hidden, optimizer=optimizer)
    t0 = time.time()
    hist = model.fit(
        X_train, to_categorical(y_train, 10),
        epochs=epochs, batch_size=BATCH_SIZE, verbose=0,
        validation_data=(X_test, to_categorical(y_test, 10)) if track_test else None,
    )
    fit_time = time.time() - t0
    test_loss, test_acc = model.evaluate(X_test, to_categorical(y_test, 10), verbose=0)
    result = {
        "train_acc": float(hist.history["accuracy"][-1]),
        "train_loss": float(hist.history["loss"][-1]),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "fit_seconds": round(fit_time, 1),
        "params": int(model.count_params()),
    }
    return model, result, hist.history


def agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Среднее и std по повторам (seed) для каждого значения фактора."""
    out = df.groupby(by, sort=False).agg(
        test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
        train_acc_mean=("train_acc", "mean"),
        test_loss_mean=("test_loss", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
        params=("params", "first"),
    ).reset_index()
    out["test_acc_std"] = out["test_acc_std"].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# 3. Базовая модель из методички
# ---------------------------------------------------------------------------
def baseline(X, y, epochs: int) -> None:
    X_train, X_test, y_train, y_test = split(X, y, TEST_SIZE, 42)
    print(f"\n[base] train {X_train.shape}, test {X_test.shape}")

    # Примеры картинок
    fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(str(y[i])); ax.axis("off")
    fig.suptitle("MNIST examples (label above image)")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "01_mnist_examples.png", dpi=150)
    plt.close(fig)

    rows = []
    best = None
    for opt in ("sgd", "adam"):
        model, res, _ = train_eval(X_train, y_train, X_test, y_test,
                                   optimizer=opt, epochs=epochs)
        res["optimizer"] = opt
        rows.append(res)
        print(f"[base] {opt:>4}: test_acc={res['test_acc']:.4f} "
              f"test_loss={res['test_loss']:.4f} ({res['fit_seconds']} c)")
        if opt == "adam":
            best = model
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "02_baseline.csv", index=False,
                              encoding="utf-8-sig")
    best.summary(print_fn=lambda s, **kw: print("        " + s))

    # Прогноз для 10 последних картинок теста (как в методичке)
    proba = best.predict(X_test[-10:], verbose=0)
    pred = np.argmax(proba, axis=1)
    print("[base] прогноз  :", pred)
    print("[base] истина   :", y_test[-10:])
    pd.DataFrame({
        "true": y_test[-10:], "pred": pred,
        "p_max": proba.max(axis=1).round(4),
    }).to_csv(RESULTS_DIR / "03_last10_predictions.csv", index=False)
    np.savetxt(RESULTS_DIR / "03_first_softmax_vector.txt", proba[0], fmt="%.6f")

    # Матрица ошибок и отчёт по классам
    y_pred = np.argmax(best.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit")
    ax.set_title("Confusion matrix (baseline, Adam)")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "04_confusion_matrix.png", dpi=150)
    plt.close(fig)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    pd.DataFrame(report).T.to_csv(RESULTS_DIR / "04_classification_report.csv",
                                  encoding="utf-8-sig")

    # Примеры ошибок
    wrong = np.where(y_pred != y_test)[0][:12]
    fig, axes = plt.subplots(2, 6, figsize=(10, 4))
    for ax, idx in zip(axes.flat, wrong):
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray_r")
        ax.set_title(f"true {y_test[idx]} / pred {y_pred[idx]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Misclassified test images")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "05_errors.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. ЗАДАНИЕ 1. Число скрытых слоёв 1..5
# ---------------------------------------------------------------------------
def task1_layers(X, y, seeds, epochs) -> pd.DataFrame:
    print("\n" + "=" * 70 + "\nЗАДАНИЕ 1. Число скрытых слоёв 1..5 "
          f"({NEURONS} нейронов, {ACTIVATION}, {epochs} эпох)\n" + "=" * 70)
    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = split(X, y, TEST_SIZE, seed)
        for opt in ("adam", "sgd"):
            for n in range(1, 6):
                _, res, _ = train_eval(X_train, y_train, X_test, y_test,
                                       n_hidden=n, optimizer=opt, seed=seed,
                                       epochs=epochs)
                rows.append({"seed": seed, "optimizer": opt, "hidden_layers": n, **res})
                print(f"  seed={seed:<5} {opt:<5} layers={n}  "
                      f"test_acc={res['test_acc']:.4f}  train_acc={res['train_acc']:.4f}  "
                      f"({res['fit_seconds']} c)")
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / "task1_layers_raw.csv", index=False, encoding="utf-8-sig")
    table = agg(raw, ["optimizer", "hidden_layers"])
    table.to_csv(RESULTS_DIR / "task1_layers.csv", index=False, encoding="utf-8-sig")
    print(table.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, opt, color in zip(axes, ("adam", "sgd"), ("tab:blue", "tab:orange")):
        t = table[table.optimizer == opt]
        ax.errorbar(t.hidden_layers, t.test_acc_mean, yerr=t.test_acc_std,
                    marker="o", capsize=4, color=color, label="test accuracy")
        ax.plot(t.hidden_layers, t.train_acc_mean, "s--", color="0.4",
                label="train accuracy")
        for xv, yv in zip(t.hidden_layers, t.test_acc_mean):
            ax.annotate(f"{yv:.4f}", (xv, yv), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Number of hidden layers (64 neurons each)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Optimizer: {opt.upper()}")
        ax.legend(loc="lower left")
    fig.suptitle("Task 1. Accuracy vs number of hidden layers (mean ± std over seeds)")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "06_task1_layers.png", dpi=150)
    plt.close(fig)
    return table


# ---------------------------------------------------------------------------
# 5. ЗАДАНИЕ 2. Размер тестовой выборки 5%..50%
# ---------------------------------------------------------------------------
def task2_test_size(X, y, seeds, epochs) -> pd.DataFrame:
    print("\n" + "=" * 70 + "\nЗАДАНИЕ 2. Доля тестовой выборки 5%..50%\n" + "=" * 70)
    sizes = [round(0.05 * k, 2) for k in range(1, 11)]
    rows = []
    for seed in seeds:
        for ts in sizes:
            X_train, X_test, y_train, y_test = split(X, y, ts, seed)
            _, res, _ = train_eval(X_train, y_train, X_test, y_test,
                                   optimizer="adam", seed=seed, epochs=epochs)
            rows.append({"seed": seed, "test_size": ts, "n_train": len(y_train),
                         "n_test": len(y_test), **res})
            print(f"  seed={seed:<5} test={ts:.0%}  n_train={len(y_train):>5}  "
                  f"n_test={len(y_test):>5}  test_acc={res['test_acc']:.4f}  "
                  f"train_acc={res['train_acc']:.4f}")
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / "task2_test_size_raw.csv", index=False, encoding="utf-8-sig")
    table = agg(raw, ["test_size"])
    table.insert(1, "n_train", raw.groupby("test_size")["n_train"].first().values)
    table.insert(2, "n_test", raw.groupby("test_size")["n_test"].first().values)
    table.to_csv(RESULTS_DIR / "task2_test_size.csv", index=False, encoding="utf-8-sig")
    print(table.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    pct = table.test_size * 100
    ax.errorbar(pct, table.test_acc_mean, yerr=table.test_acc_std, marker="o",
                capsize=4, label="test accuracy (mean ± std)")
    ax.plot(pct, table.train_acc_mean, "s--", color="0.4", label="train accuracy")
    for xv, yv in zip(pct, table.test_acc_mean):
        ax.annotate(f"{yv:.4f}", (xv, yv), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8)
    ax2 = ax.twinx()
    ax2.bar(pct, table.n_train, width=2.5, alpha=0.15, color="tab:green",
            label="train size")
    ax2.set_ylabel("Training set size", color="tab:green")
    ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_xticks(pct)
    ax.set_xlabel("Test set share, %")
    ax.set_ylabel("Accuracy")
    ax.set_title("Task 2. Accuracy vs test-set size (2x64 sigmoid, Adam)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "07_task2_test_size.png", dpi=150)
    plt.close(fig)
    return table


# ---------------------------------------------------------------------------
# 6. ЗАДАНИЕ 3. Оптимизаторы
# ---------------------------------------------------------------------------
OPTIMIZERS = ["sgd", "sgd_momentum", "adagrad", "adadelta",
              "rmsprop", "adam", "nadam", "adamax"]


def task3_optimizers(X, y, seeds, epochs) -> pd.DataFrame:
    print("\n" + "=" * 70 + "\nЗАДАНИЕ 3. Оптимизаторы\n" + "=" * 70)
    rows, curves = [], []
    for seed in seeds:
        X_train, X_test, y_train, y_test = split(X, y, TEST_SIZE, seed)
        for opt in OPTIMIZERS:
            _, res, hist = train_eval(X_train, y_train, X_test, y_test,
                                      optimizer=opt, seed=seed, epochs=epochs,
                                      track_test=True)
            rows.append({"seed": seed, "optimizer": opt, **res})
            for ep in range(epochs):
                curves.append({"seed": seed, "optimizer": opt, "epoch": ep + 1,
                               "loss": hist["loss"][ep], "accuracy": hist["accuracy"][ep],
                               "val_loss": hist["val_loss"][ep],
                               "val_accuracy": hist["val_accuracy"][ep]})
            print(f"  seed={seed:<5} {opt:<13} test_acc={res['test_acc']:.4f}  "
                  f"test_loss={res['test_loss']:.4f}  ({res['fit_seconds']} c)")
    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS_DIR / "task3_optimizers_raw.csv", index=False, encoding="utf-8-sig")
    curves = pd.DataFrame(curves)
    curves.to_csv(RESULTS_DIR / "task3_curves.csv", index=False, encoding="utf-8-sig")
    table = agg(raw, ["optimizer"]).sort_values("test_acc_mean", ascending=False)
    table.to_csv(RESULTS_DIR / "task3_optimizers.csv", index=False, encoding="utf-8-sig")
    print(table.round(4).to_string(index=False))

    mean_curves = curves.groupby(["optimizer", "epoch"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = plt.get_cmap("tab10").colors
    for i, opt in enumerate(OPTIMIZERS):
        c = mean_curves[mean_curves.optimizer == opt]
        axes[0].plot(c.epoch, c.loss, marker="o", ms=3, color=palette[i], label=opt)
        axes[1].plot(c.epoch, c.val_accuracy, marker="o", ms=3, color=palette[i], label=opt)
    axes[0].set_title("Training loss (categorical cross-entropy)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_yscale("log")
    axes[1].set_title("Test accuracy after each epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(ncol=2, fontsize=9)
    for ax in axes:
        ax.set_xticks(range(1, epochs + 1))
    fig.suptitle("Task 3. Learning curves for different optimizers (mean over seeds)")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "08_task3_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    t = table.sort_values("test_acc_mean")
    bars = ax.barh(t.optimizer, t.test_acc_mean, xerr=t.test_acc_std, capsize=3,
                   color=[palette[OPTIMIZERS.index(o)] for o in t.optimizer])
    for bar, v in zip(bars, t.test_acc_mean):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Test accuracy")
    ax.set_title(f"Task 3. Final test accuracy after {epochs} epochs")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "09_task3_bars.png", dpi=150)
    plt.close(fig)
    return table


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="быстрый прогон: 1 seed, 3 эпохи")
    parser.add_argument("--only", choices=["base", "t1", "t2", "t3"], default=None)
    args = parser.parse_args()

    seeds = SEEDS[:1] if args.quick else SEEDS
    epochs = 3 if args.quick else EPOCHS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 70)
    print("ЛАБОРАТОРНАЯ РАБОТА №4. ВВЕДЕНИЕ В НЕЙРОННЫЕ СЕТИ (MNIST)")
    print(f"TensorFlow {tf.__version__}, seeds={seeds}, epochs={epochs}")
    print("=" * 70)

    single_neuron_demo()
    X, y = load_mnist()

    if args.only in (None, "base"):
        baseline(X, y, epochs)
    if args.only in (None, "t1"):
        task1_layers(X, y, seeds, epochs)
    if args.only in (None, "t2"):
        task2_test_size(X, y, seeds, epochs)
    if args.only in (None, "t3"):
        task3_optimizers(X, y, seeds, epochs)

    print(f"\nГОТОВО за {(time.time() - t0) / 60:.1f} мин. Результаты: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
