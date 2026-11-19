# -*- coding: utf-8 -*-
"""
================================================================================
 Лабораторная работа №3. Регрессионный анализ временных рядов
 Прогнозирование энергопотребления (OPSD Germany daily)
================================================================================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ
--------------------
1. Скачивает набор данных (если его нет) и загружает его в pandas.
2. Строит базовые графики: ряд энергопотребления, скользящие средние, АКФ.
3. Демонстрирует корректировку стационарности из методички (выравнивание ряда).
4. Готовит временные (календарные) признаки и обучает 5 регрессионных моделей:
      LR  - LinearRegression
      NN  - MLPRegressor (нейронная сеть)
      KNN - KNeighborsRegressor
      RF  - RandomForestRegressor
      SVR - SVR (машина опорных векторов)
5. ЗАДАНИЕ 2: для каждого горизонта прогноза от 1 до 12 месяцев подбирает
   тестовую выборку как последние N месяцев ряда, обучает модели на всей
   предшествующей истории и считает R2, MSE, RMSE, MAE. Результат сохраняется
   в таблицы (CSV/XLSX/Markdown) и на графики зависимостей метрик от горизонта.
6. ЗАДАНИЕ 3: выбирает лучшую модель, обучает её на 2006-2016 гг. и делает
   прогноз на весь 2017 год. Строит график «реальность vs прогноз», диаграмму
   рассеяния, график остатков и сохраняет таблицу прогнозов.

ЗАПУСК
------
    .venv\\Scripts\\python.exe lab3_solution.py

Все результаты (графики и таблицы) складываются в папку results/.
Скрипт не требует интерактивного окна: графики сохраняются в PNG.
"""

from __future__ import annotations

import sys
import time
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Backend Agg позволяет строить графики без открытия окон (важно для скрипта).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Подавляем технические предупреждения sklearn/pandas, чтобы вывод был читаемым.
warnings.filterwarnings("ignore")

# На Windows консоль может быть в cp1251. Переключаем stdout на UTF-8,
# чтобы русские сообщения не вызывали UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Пути и константы
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DATASET_PATH = DATA_DIR / "opsd_germany_daily.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
)

TRAIN_END = "2016-12-31"       # конец обучения для итогового прогноза
TEST_START = "2017-01-01"      # начало тестового года
TEST_END = "2017-12-31"
RANDOM_STATE = 42

RUN_CV = True                  # запускать ли дополнительную кросс-валидацию
CV_SPLITS = 5                  # число фолдов TimeSeriesSplit

# Единый стиль графиков.
sns.set_theme(style="whitegrid")
plt.rcParams["figure.max_open_warning"] = 0


# ---------------------------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------------------------
def ensure_dataset() -> Path:
    """Скачивает CSV, если локальной копии ещё нет, и возвращает путь к файлу."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_PATH.exists():
        print("[data] Локальный файл не найден. Скачиваю датасет...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        print(f"[data] Сохранён: {DATASET_PATH}")
    else:
        print(f"[data] Использую локальный файл: {DATASET_PATH}")
    return DATASET_PATH


def load_data() -> pd.DataFrame:
    """
    Читает датасет и делает столбец Date индексом типа DatetimeIndex.

    В методичке дата и время сначала склеивались вручную. Здесь достаточно
    parse_dates=['Date']: для дневного ряда время 00:00:00 не несёт информации.
    """
    path = ensure_dataset()
    data = pd.read_csv(path, parse_dates=["Date"])
    data = data.sort_values("Date").set_index("Date")
    print(
        f"[data] Период: {data.index.min().date()} .. {data.index.max().date()}, "
        f"строк: {len(data)}"
    )
    return data


# ---------------------------------------------------------------------------
# 2. Разведочный анализ (EDA)
# ---------------------------------------------------------------------------
def plot_eda(data: pd.DataFrame) -> None:
    """Строит ряд энергопотребления со скользящими средними и АКФ."""
    # --- Ряд + тренды -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        data.index, data["Consumption"],
        marker=".", markersize=2, linestyle="None", color="0.6",
        label="Daily consumption",
    )
    ax.plot(
        data.index, data["Consumption"].rolling(7, center=True).mean(),
        linewidth=1.5, label="7-day rolling mean",
    )
    ax.plot(
        data.index, data["Consumption"].rolling(365, center=True).mean(),
        linewidth=2.5, color="0.2", label="365-day rolling mean (trend)",
    )
    ax.set_title("Electricity consumption in Germany")
    ax.set_xlabel("Year")
    ax.set_ylabel("Consumption (GWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "01_consumption_and_trends.png", dpi=150)
    plt.close(fig)

    # --- Автокорреляционная функция --------------------------------------
    # АКФ показывает, насколько текущее значение связано с прошлыми.
    # Строим по двухлетнему окну, как в методичке, иначе график слишком плотный.
    fig, ax = plt.subplots(figsize=(12, 4))
    pd.plotting.autocorrelation_plot(
        data.loc["2012-01":"2013-12", "Consumption"], ax=ax
    )
    ax.set_title("Autocorrelation of daily consumption (2012-2013)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "02_autocorrelation.png", dpi=150)
    plt.close(fig)
    print("[eda] Сохранены графики: 01_consumption_and_trends.png, "
          "02_autocorrelation.png")


# ---------------------------------------------------------------------------
# 3. Демонстрация корректировки стационарности из методички
# ---------------------------------------------------------------------------
def stationarity_correction_demo(data: pd.DataFrame) -> None:
    """
    Повторяет блок методички «Корректировки стационарности временного ряда».

    Идея: ряд потребления имеет сезонный тренд. Чтобы сделать его более
    стационарным, к каждому значению прибавляется поправка
        trend_residuals = max(365-дневный тренд) - 365-дневный тренд.
    Там, где тренд ниже максимума, поправка больше; там, где тренд максимален,
    поправка равна нулю. В результате амплитуда ряда выравнивается.

    Важно: это демонстрация из методички. В основных заданиях 2 и 3 мы
    прогнозируем исходное (реальное) энергопотребление, а не выровненное.
    """
    columns = ["Consumption", "Wind", "Solar", "Wind+Solar"]

    # Скользящие средние. center=True ставит среднее в центр окна.
    data_365 = data[columns].rolling(window=365, center=True).mean()
    data_7 = data[columns].rolling(window=7, center=True).mean()

    # На краях окна 365 дней появляются NaN. Заполняем их соседними значениями.
    # .bfill().ffill() — современная замена устаревшему fillna(method=...).
    data_365["Consumption"] = data_365["Consumption"].bfill().ffill()

    trend_residuals = data_365["Consumption"].max() - data_365["Consumption"]
    consumption_eq = data["Consumption"] + trend_residuals
    trend_eq = data_365["Consumption"] + trend_residuals * 2
    consumption_eq_roll_7 = consumption_eq.rolling(7, center=True).mean()

    # --- График выравнивания --------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(data.index, data["Consumption"], ".", markersize=3,
            linestyle="None", label="Consumption")
    ax.plot(consumption_eq.index, consumption_eq, ".", markersize=2,
            color="0", linestyle="None", label="Consumption equalized")
    ax.plot(consumption_eq_roll_7.index, consumption_eq_roll_7, linewidth=2,
            label="7-day rolling mean, equalized")
    ax.plot(data_7.index, data_7["Consumption"], linewidth=1.5,
            label="7-day rolling mean, original")
    ax.plot(trend_eq.index, trend_eq, color="0.2", linewidth=3,
            label="365-day trend, equalized")
    ax.axhline(1600, linewidth=1, linestyle="--", color="tab:red",
               label="constant = 1600")
    ax.set_title("Stationarity correction: original vs equalized series")
    ax.set_xlabel("Year")
    ax.set_ylabel("Consumption (GWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_stationarity_correction.png", dpi=150)
    plt.close(fig)

    # --- Среднее и стандартное отклонение по годам ----------------------
    years = sorted(data.index.year.unique())
    stats = pd.DataFrame(
        index=years,
        columns=["raw_mean", "raw_std", "eq_mean", "eq_std"],
        dtype=float,
    )
    for year in years:
        stats.loc[year, "raw_mean"] = data_7.loc[str(year), "Consumption"].mean()
        stats.loc[year, "raw_std"] = data_7.loc[str(year), "Consumption"].std()
        stats.loc[year, "eq_mean"] = consumption_eq_roll_7.loc[str(year)].mean()
        stats.loc[year, "eq_std"] = consumption_eq_roll_7.loc[str(year)].std()

    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()
    ax1.plot(stats.index, stats["raw_mean"], "--", label="raw mean")
    ax1.plot(stats.index, stats["eq_mean"], "--", label="equalized mean")
    ax2.plot(stats.index, stats["raw_std"], label="raw std")
    ax2.plot(stats.index, stats["eq_std"], label="equalized std")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean")
    ax2.set_ylabel("Std (variance proxy)")
    ax1.set_title("Yearly mean and standard deviation: before vs after correction")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_stationarity_yearly_stats.png", dpi=150)
    plt.close(fig)

    stats.round(2).to_csv(RESULTS_DIR / "stationarity_yearly_stats.csv",
                          encoding="utf-8-sig")
    print("[stationarity] Сохранены 03_stationarity_correction.png, "
          "04_stationarity_yearly_stats.png и stationarity_yearly_stats.csv")


# ---------------------------------------------------------------------------
# 4. Признаки и модели
# ---------------------------------------------------------------------------
def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет к датафрейму календарные и гармонические (Fourier) признаки.

    Почему именно такие признаки:
      * dayofweek / month / quarter / dayofyear / day / weekofyear — базовые
        календарные признаки из методички; они описывают сезонность.
      * sin/cos годовых гармоник (1-3) — гладкое циклическое представление
        сезонности. В отличие от «номера дня года» оно не имеет разрыва
        между 31 декабря и 1 января и хорошо работает для линейных/ядерных
        моделей.
      * sin/cos недельной гармоники — учитывают недельный цикл (будни/выходные).
      * is_weekend — простая и интерпретируемая бинарная поправка.

    Чего здесь СОЗНАТЕЛЬНО нет:
      * лагов (Yesterday, Yesterday_Diff). В методичке они используются, но
        это прогноз «на один шаг вперёд»: для тестового дня берётся реальное
        потребление вчера. Для честного многомесячного прогноза такие признаки
        были бы утечкой данных. Мы прогнозируем весь год только по календарю.
      * year как признак: деревья не умеют экстраполировать за пределы
        обучающего диапазона, поэтому числовой год может только навредить.
    """
    out = df.copy()
    idx = out.index

    out["dayofweek"] = idx.dayofweek
    out["quarter"] = idx.quarter
    out["month"] = idx.month
    out["dayofyear"] = idx.dayofyear
    out["day"] = idx.day
    out["weekofyear"] = idx.isocalendar().week.astype(int)
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Годовые гармоники: 2*pi*k*t / 365.25 (365.25 учитывает високосные годы).
    day_of_year = idx.dayofyear.to_numpy()
    for k in (1, 2, 3):
        out[f"sin_year_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        out[f"cos_year_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)

    # Недельная гармоника.
    day_of_week = idx.dayofweek.to_numpy()
    out["sin_week"] = np.sin(2 * np.pi * day_of_week / 7)
    out["cos_week"] = np.cos(2 * np.pi * day_of_week / 7)

    return out


def build_models() -> list[tuple[str, object]]:
    """
    Возвращает список (имя, необученная модель) для пяти алгоритмов.

    Особенности предобработки:
      * Масштабирование признаков (StandardScaler) критично для KNN, SVR и
        нейронной сети: они чувствительны к разному масштабу признаков.
      * SVR дополнительно масштабирует целевую переменную через
        TransformedTargetRegressor: иначе при C=1 и epsilon=0.1 модель
        «не дотягивается» до значений порядка 1300 ГВт·ч и недообучается.
      * Деревья (RandomForest) и линейная регрессия в масштабировании не
        нуждаются (для линейной регрессии оно также безвредно).
    """
    nn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=800,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )),
    ])

    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5, weights="distance")),
    ])

    svr = TransformedTargetRegressor(
        regressor=Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)),
        ]),
        transformer=StandardScaler(),
    )

    return [
        ("LR", LinearRegression()),
        ("NN", nn),
        ("KNN", knn),
        ("RF", RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )),
        ("SVR", svr),
    ]


def regression_results(y_true, y_pred, model_name: str = "model",
                       verbose: bool = True) -> dict:
    """
    Считает четыре метрики качества регрессии:
        R2   - коэффициент детерминации (чем ближе к 1, тем лучше);
        MSE  - средняя квадратичная ошибка;
        RMSE - корень из MSE (в тех же единицах, что и потребление);
        MAE  - средняя абсолютная ошибка.
    Возвращает словарь с метриками (удобно складывать в таблицу).
    """
    mse = mean_squared_error(y_true, y_pred)
    result = {
        "R2": r2_score(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }
    if verbose:
        print("-" * 63)
        print(f"Модель: {model_name}")
        for key, value in result.items():
            print(f"  {key:<4}: {value:.4f}")
    return result


# ---------------------------------------------------------------------------
# 5. ЗАДАНИЕ 2. Зависимость точности от размера тестовой выборки
# ---------------------------------------------------------------------------
def evaluate_horizons(df: pd.DataFrame, horizons=range(1, 13)) -> pd.DataFrame:
    """
    Для каждого горизонта N = 1..12 месяцев:
      * тест — последние N календарных месяцев ряда (хвост данных);
      * обучение — вся история ДО начала теста;
      * все 5 моделей обучаются заново и оцениваются на тесте.

    Такой подход моделирует реальную ситуацию: чем на больший срок мы
    прогнозируем, тем меньше свежих данных доступно для обучения и тем
    длиннее участок, который нужно предсказать.

    Возвращает «длинную» таблицу со столбцами:
        model, horizon_months, train_size, test_size, R2, MSE, RMSE, MAE,
        fit_seconds
    """
    records: list[dict] = []
    last_date = pd.Timestamp(TEST_END)

    for n_months in horizons:
        # Начало последних N месяцев. Например:
        #   N=1  -> 2017-12-01
        #   N=2  -> 2017-11-01
        #   N=12 -> 2017-01-01
        test_start = (last_date + pd.Timedelta(days=1)) - pd.DateOffset(
            months=n_months
        )
        train_end = test_start - pd.Timedelta(days=1)

        train = df.loc[:train_end]
        test = df.loc[test_start:last_date]

        X_train = train.drop(columns=["Consumption"])
        y_train = train["Consumption"]
        X_test = test.drop(columns=["Consumption"])
        y_test = test["Consumption"]

        print(
            f"\n[horizon] N={n_months:>2} мес. | "
            f"train {train.index.min().date()}..{train.index.max().date()} "
            f"({len(train)} дн.) | test {test.index.min().date()}.."
            f"{test.index.max().date()} ({len(test)} дн.)"
        )

        # build_models() создаёт новые экземпляры, чтобы модели не «запоминали»
        # результаты предыдущего горизонта. clone() здесь не обязателен, но
        # оставлен для наглядности и безопасности.
        for name, base_model in build_models():
            model = clone(base_model)
            t0 = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fit_seconds = time.time() - t0

            metrics = regression_results(y_test, y_pred, name, verbose=False)
            records.append({
                "model": name,
                "horizon_months": n_months,
                "train_size": len(train),
                "test_size": len(test),
                "fit_seconds": round(fit_seconds, 2),
                **metrics,
            })
            print(
                f"    {name:<4} R2={metrics['R2']:>6.3f}  "
                f"RMSE={metrics['RMSE']:>7.2f}  "
                f"MAE={metrics['MAE']:>7.2f}  "
                f"({fit_seconds:.1f} c)"
            )

    return pd.DataFrame.from_records(records)


def save_horizon_tables(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Сохраняет длинную таблицу и сводные таблицы по каждой метрике."""
    long_df.to_csv(
        RESULTS_DIR / "metrics_vs_horizon.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = ["R2", "MSE", "RMSE", "MAE"]
    pivots: dict[str, pd.DataFrame] = {}

    # Excel-книга: отдельный лист на каждую метрику + исходные данные.
    with pd.ExcelWriter(RESULTS_DIR / "metrics_vs_horizon.xlsx") as writer:
        long_df.to_excel(writer, sheet_name="long", index=False)
        for metric in metrics:
            pivot = long_df.pivot(
                index="horizon_months", columns="model", values=metric
            )
            pivot.index.name = "horizon_months"
            pivots[metric] = pivot
            pivot.round(4).to_excel(writer, sheet_name=metric)

    # Markdown-таблицы удобно вставлять в отчёт.
    lines = [
        "# Зависимость метрик от горизонта прогноза",
        "",
        "Тест — последние N месяцев ряда, обучение — вся предшествующая история.",
        "",
    ]
    for metric in metrics:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append(pivots[metric].round(4).to_markdown())
        lines.append("")
    (RESULTS_DIR / "metrics_vs_horizon.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Красивый текстовый вывод в консоль.
    for metric in metrics:
        print(f"\n===== {metric} (строки — горизонт, столбцы — модель) =====")
        print(pivots[metric].round(4).to_string())

    return pivots


def plot_metrics_vs_horizon(long_df: pd.DataFrame) -> None:
    """Строит 4 графика: R2, RMSE, MAE, MSE в зависимости от горизонта."""
    metrics = [("R2", "R2 (higher is better)"),
               ("RMSE", "RMSE, GWh (lower is better)"),
               ("MAE", "MAE, GWh (lower is better)"),
               ("MSE", "MSE (lower is better)")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = plt.get_cmap("tab10").colors

    for ax, (metric, title) in zip(axes.flat, metrics):
        pivot = long_df.pivot(
            index="horizon_months", columns="model", values=metric
        )
        for i, column in enumerate(pivot.columns):
            ax.plot(
                pivot.index, pivot[column],
                marker="o", linewidth=1.8, color=colors[i], label=column,
            )
        ax.set_title(title)
        ax.set_xlabel("Forecast horizon, months")
        ax.set_ylabel(metric)
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Model accuracy vs test-set size (forecast horizon)", y=1.06
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "05_metrics_vs_horizon.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("\n[plot] Сохранён график 05_metrics_vs_horizon.png")


# ---------------------------------------------------------------------------
# 6. Дополнительно: кросс-валидация на обучающей части (как в методичке)
# ---------------------------------------------------------------------------
def run_cv_comparison(df: pd.DataFrame) -> None:
    """
    TimeSeriesSplit разбивает обучающую часть на последовательные фолды:
    обучение всегда на прошлом, валидация — на следующем по времени участке.
    Это корректнее обычного KFold, который перемешивает время.

    Результат — boxplot распределения R2 по фолдам для пяти моделей.
    """
    print(f"\n[cv] Кросс-валидация TimeSeriesSplit (n_splits={CV_SPLITS})...")
    train = df.loc[:TRAIN_END]
    X = train.drop(columns=["Consumption"])
    y = train["Consumption"]

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores: list[np.ndarray] = []
    names: list[str] = []

    for name, model in build_models():
        t0 = time.time()
        cv_scores = cross_val_score(
            model, X, y, cv=tscv, scoring="r2", n_jobs=1
        )
        scores.append(cv_scores)
        names.append(name)
        print(
            f"    {name:<4} R2 mean={cv_scores.mean():.4f} "
            f"std={cv_scores.std():.4f} ({time.time() - t0:.1f} c)"
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(scores, labels=names)
    ax.set_title("Algorithm comparison: R2 over TimeSeriesSplit folds")
    ax.set_ylabel("R2")
    ax.set_xlabel("Model")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "06_cv_algorithm_comparison.png", dpi=150)
    plt.close(fig)

    # Сохраняем сами значения R2 по фолдам.
    cv_table = pd.DataFrame(
        {name: series for name, series in zip(names, scores)}
    )
    cv_table.to_csv(RESULTS_DIR / "cv_scores.csv", index=False,
                    encoding="utf-8-sig")
    print("[cv] Сохранены 06_cv_algorithm_comparison.png и cv_scores.csv")


# ---------------------------------------------------------------------------
# 7. ЗАДАНИЕ 3. Итоговый прогноз на 1 год
# ---------------------------------------------------------------------------
def summarize_models(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Сводка по моделям: средние метрики по всем горизонтам.
    Используется для автоматического выбора лучшей модели.
    """
    summary = (
        long_df.groupby("model")
        .agg(
            mean_R2=("R2", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_MAE=("MAE", "mean"),
            mean_MSE=("MSE", "mean"),
        )
        .sort_values("mean_R2", ascending=False)
    )
    summary.round(4).to_csv(
        RESULTS_DIR / "models_summary.csv", encoding="utf-8-sig"
    )
    return summary


def get_feature_importances(model, feature_names) -> pd.Series | None:
    """
    Достаёт важность признаков из обученной модели.
    Работает для деревьев (feature_importances_) и линейных моделей (|coef_|).
    Для KNN/SVR важности не определены — возвращается None.
    """
    estimator = model
    if isinstance(estimator, TransformedTargetRegressor):
        estimator = estimator.regressor_
    if isinstance(estimator, Pipeline):
        estimator = estimator.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
    else:
        return None

    if len(values) != len(feature_names):
        return None

    return (
        pd.Series(np.abs(values), index=feature_names)
        .sort_values(ascending=False)
    )


def final_forecast(df: pd.DataFrame, model_name: str) -> None:
    """
    Обучает выбранную модель на 2006-2016 гг. и прогнозирует весь 2017 год.
    Строит и сохраняет графики сравнения, рассеяния и остатков.
    """
    train = df.loc[:TRAIN_END]
    test = df.loc[TEST_START:TEST_END]

    X_train = train.drop(columns=["Consumption"])
    y_train = train["Consumption"]
    X_test = test.drop(columns=["Consumption"])
    y_test = test["Consumption"]

    models = dict(build_models())
    model = clone(models[model_name])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "=" * 63)
    print(f"ИТОГОВЫЙ ПРОГНОЗ НА 2017 ГОД. Модель: {model_name}")
    print("=" * 63)
    metrics = regression_results(y_test, y_pred, model_name, verbose=True)

    # --- График «реальность vs прогноз» ---------------------------------
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(test.index, y_test, linewidth=2, label="Real")
    ax.plot(test.index, y_pred, linewidth=2, label="Prediction")
    ax.set_title(f"2017 forecast: {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption (GWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "07_final_forecast.png", dpi=150)
    plt.close(fig)

    # --- Диаграмма рассеяния --------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, s=12, alpha=0.6)
    limits = [
        min(y_test.min(), y_pred.min()) - 20,
        max(y_test.max(), y_pred.max()) + 20,
    ]
    ax.plot(limits, limits, "--", color="tab:red", label="Ideal y = x")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Real consumption (GWh)")
    ax.set_ylabel("Predicted consumption (GWh)")
    ax.set_title(f"Scatter: {model_name}, R2={metrics['R2']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "08_final_scatter.png", dpi=150)
    plt.close(fig)

    # --- Остатки --------------------------------------------------------
    residuals = y_test.to_numpy() - y_pred
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(residuals, bins=40, edgecolor="white")
    ax.axvline(0, color="tab:red", linestyle="--")
    ax.set_title(f"Residuals: {model_name}")
    ax.set_xlabel("Real - Prediction (GWh)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "09_final_residuals.png", dpi=150)
    plt.close(fig)

    # --- Таблица прогнозов ----------------------------------------------
    forecast_table = pd.DataFrame({
        "Date": test.index,
        "Real": y_test.to_numpy(),
        "Prediction": y_pred,
        "Error": residuals,
    })
    forecast_table.to_csv(
        RESULTS_DIR / "final_forecast_2017.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --- Важность признаков лучшей модели (если определена) -------------
    importances = get_feature_importances(model, X_train.columns)
    if importances is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        importances.head(12).sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Feature importance ({model_name})")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "10_feature_importance.png", dpi=150)
        plt.close(fig)
        print("[final] Сохранён график важности признаков.")

    # Дополнительно: важность признаков RandomForest как интерпретируемой
    # модели, даже если лучшей оказалась другая.
    rf = RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_importances = pd.Series(
        rf.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    rf_importances.head(12).sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature importance (RandomForest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "11_rf_feature_importance.png", dpi=150)
    plt.close(fig)

    print("[final] Сохранены 07_final_forecast.png, 08_final_scatter.png, "
          "09_final_residuals.png, 10/11_feature_importance*.png, "
          "final_forecast_2017.csv")


# ---------------------------------------------------------------------------
# 8. main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 63)
    print("ЛАБОРАТОРНАЯ РАБОТА №3. РЕГРЕССИОННЫЙ АНАЛИЗ ВРЕМЕННЫХ РЯДОВ")
    print("=" * 63)

    # 1) Данные и EDA.
    data = load_data()
    plot_eda(data)

    # 2) Демонстрация корректировки стационарности из методички.
    stationarity_correction_demo(data)

    # 3) Признаки. Работаем только с Consumption: остальные столбцы (ветер,
    #    солнце) не нужны для прогноза потребления и содержат NaN.
    df = make_time_features(data[["Consumption"]])

    # 4) Дополнительная кросс-валидация (как в блокноте).
    if RUN_CV:
        run_cv_comparison(df)

    # 5) ЗАДАНИЕ 2: метрики при разном размере тестовой выборки.
    print("\n" + "=" * 63)
    print("ЗАДАНИЕ 2. Зависимость точности от горизонта 1..12 месяцев")
    print("=" * 63)
    long_df = evaluate_horizons(df)
    save_horizon_tables(long_df)
    plot_metrics_vs_horizon(long_df)

    # 6) ЗАДАНИЕ 3: лучшая модель и прогноз на 1 год.
    print("\n" + "=" * 63)
    print("ЗАДАНИЕ 3. Итоговый прогноз на 1 год")
    print("=" * 63)
    summary = summarize_models(long_df)
    print("\nСводка средних метрик по всем горизонтам:")
    print(summary.round(4).to_string())

    best_model_name = summary.index[0]
    print(f"\nЛучшая модель по среднему R2: {best_model_name}")
    final_forecast(df, best_model_name)

    print("\n" + "=" * 63)
    print(f"ГОТОВО за {time.time() - t_start:.1f} c.")
    print(f"Все результаты находятся в папке: {RESULTS_DIR}")
    print("=" * 63)


if __name__ == "__main__":
    main()
