# -*- coding: utf-8 -*-
"""
Генератор PDF-отчёта по лабораторной работе №3.

Скрипт читает готовые результаты из папки results/ (таблицы CSV и PNG-графики),
которые создаёт lab3_solution.py, и собирает единый PDF-отчёт в папке
report_pdf/. Если папки results/ нет, сначала запустите lab3_solution.py.

Запуск:
    .venv\\Scripts\\python.exe make_pdf_report.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
REPORT_DIR = BASE_DIR / "report_pdf"
REPORT_PATH = REPORT_DIR / "Лабораторная_работа_3.pdf"

FONT_DIR = (
    BASE_DIR / ".venv" / "Lib" / "site-packages" / "matplotlib"
    / "mpl-data" / "fonts" / "ttf"
)

# ---------------------------------------------------------------------------
# Шрифты с поддержкой кириллицы
# ---------------------------------------------------------------------------
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(
    TTFont("DejaVu-BoldOblique", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf"))
)
registerFontFamily(
    "DejaVu",
    normal="DejaVu",
    bold="DejaVu-Bold",
    italic="DejaVu-Oblique",
    boldItalic="DejaVu-BoldOblique",
)

# ---------------------------------------------------------------------------
# Стили
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "TitleRU", parent=styles["Title"], fontName="DejaVu-Bold",
    fontSize=22, leading=28, alignment=TA_CENTER, spaceAfter=6,
)
SUBTITLE = ParagraphStyle(
    "SubtitleRU", parent=styles["Normal"], fontName="DejaVu",
    fontSize=13, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
)
H1 = ParagraphStyle(
    "H1RU", parent=styles["Heading1"], fontName="DejaVu-Bold",
    fontSize=15, leading=20, spaceBefore=10, spaceAfter=6,
    textColor=colors.HexColor("#1f3b57"),
)
H2 = ParagraphStyle(
    "H2RU", parent=styles["Heading2"], fontName="DejaVu-Bold",
    fontSize=12, leading=16, spaceBefore=8, spaceAfter=4,
    textColor=colors.HexColor("#2f5d86"),
)
BODY = ParagraphStyle(
    "BodyRU", parent=styles["BodyText"], fontName="DejaVu",
    fontSize=10, leading=15, alignment=TA_JUSTIFY, spaceAfter=5,
)
BULLET = ParagraphStyle(
    "BulletRU", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3,
)
CAPTION = ParagraphStyle(
    "CaptionRU", parent=styles["Normal"], fontName="DejaVu-Oblique",
    fontSize=8.5, leading=11, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), spaceBefore=3, spaceAfter=8,
)
SMALL = ParagraphStyle(
    "SmallRU", parent=BODY, fontSize=8.5, leading=12, textColor=colors.HexColor("#555555"),
)
CENTER = ParagraphStyle(
    "CenterRU", parent=BODY, alignment=TA_CENTER,
)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def p(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str) -> Paragraph:
    return Paragraph(text, BULLET, bulletText="•")


def make_image(filename: str, max_width=168 * mm, max_height=115 * mm) -> Image:
    """Вставляет PNG с сохранением пропорций и ограничением размеров."""
    path = RESULTS_DIR / filename
    with PILImage.open(path) as image:
        width_px, height_px = image.size
    ratio = height_px / width_px

    width = max_width
    height = width * ratio
    if height > max_height:
        height = max_height
        width = height / ratio
    return Image(str(path), width=width, height=height)


def figure(filename: str, caption: str, max_height=115 * mm) -> KeepTogether:
    """Картинка + подпись, которые не должны разрываться между страницами."""
    return KeepTogether([
        make_image(filename, max_height=max_height),
        p(caption, CAPTION),
    ])


def styled_table(
    data,
    col_widths=None,
    font_size=8.5,
    align_first_left=True,
    header_bg="#23445f",
) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a9b6c2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef3f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if align_first_left:
        commands.append(("ALIGN", (0, 0), (0, -1), "LEFT"))
    table.setStyle(TableStyle(commands))
    return table


def pivot_to_table(pivot: pd.DataFrame, fmt: str, title: str) -> Table:
    """Превращает pivot-таблицу pandas в отчётную таблицу reportlab."""
    data = [[title] + [str(c) for c in pivot.columns]]
    for index, row in pivot.iterrows():
        data.append([str(int(index))] + [fmt.format(value) for value in row.values])
    return styled_table(data, col_widths=[30 * mm] + [27 * mm] * len(pivot.columns))


def on_page(canvas, doc):
    """Колонтитулы на каждой странице."""
    canvas.saveState()
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(
        18 * mm, 10 * mm,
        "Лабораторная работа №3. Регрессионный анализ временных рядов",
    )
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"стр. {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#c8d2dc"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Данные для отчёта
# ---------------------------------------------------------------------------
def load_results():
    long_df = pd.read_csv(
        RESULTS_DIR / "metrics_vs_horizon.csv", encoding="utf-8-sig"
    )
    summary = pd.read_csv(
        RESULTS_DIR / "models_summary.csv", index_col=0, encoding="utf-8-sig"
    )
    forecast = pd.read_csv(
        RESULTS_DIR / "final_forecast_2017.csv",
        parse_dates=["Date"],
        encoding="utf-8-sig",
    )

    model_order = ["LR", "NN", "KNN", "RF", "SVR"]
    pivots = {}
    for metric in ["R2", "MSE", "RMSE", "MAE"]:
        pivot = long_df.pivot(
            index="horizon_months", columns="model", values=metric
        )
        pivot = pivot.reindex(columns=[m for m in model_order if m in pivot.columns])
        pivots[metric] = pivot

    return long_df, summary, forecast, pivots


def final_metrics(forecast: pd.DataFrame) -> dict:
    y_true = forecast["Real"].to_numpy()
    y_pred = forecast["Prediction"].to_numpy()
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "mean_real": float(np.mean(y_true)),
    }


# ---------------------------------------------------------------------------
# Сборка отчёта
# ---------------------------------------------------------------------------
def build_report():
    long_df, summary, forecast, pivots = load_results()
    metrics = final_metrics(forecast)

    best_model = summary.index[0]
    best_mean_r2 = float(summary.loc[best_model, "mean_R2"])
    best_horizon_series = pivots["R2"].stack()
    (best_horizon, best_horizon_model) = best_horizon_series.idxmax()
    best_horizon_r2 = float(best_horizon_series.max())

    mean_real = metrics["mean_real"]
    rel_rmse = metrics["RMSE"] / mean_real * 100

    # Помесячные средние для наглядной таблицы итогового прогноза.
    forecast = forecast.copy()
    forecast["Month"] = forecast["Date"].dt.strftime("%Y-%m")
    monthly = (
        forecast.groupby("Month")
        .agg(Real=("Real", "mean"), Prediction=("Prediction", "mean"))
        .reset_index()
    )
    monthly["AbsError"] = (monthly["Real"] - monthly["Prediction"]).abs()

    story = []

    # ======================= Титульная страница =======================
    story.append(Spacer(1, 28 * mm))
    story.append(p("ЛАБОРАТОРНАЯ РАБОТА №3", TITLE))
    story.append(Spacer(1, 4 * mm))
    story.append(p("Регрессионный анализ временных рядов", SUBTITLE))
    story.append(Spacer(1, 3 * mm))
    story.append(p("Прогнозирование энергопотребления (OPSD Germany daily)", SUBTITLE))
    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(width="60%", thickness=1.2,
                            color=colors.HexColor("#23445f"), hAlign="CENTER"))
    story.append(Spacer(1, 10 * mm))

    info = [
        ["Выполнил:", "Ильин А.А."],
        ["Группа:", "УИТСм-11"],
        ["Дата:", "06.11.2026"],
    ]
    # Для титульной таблицы создаём Table напрямую: styled_table всегда
    # оформляет первую строку как тёмную шапку, из-за чего текст в ней
    # становился белым на белом фоне.
    info_table = Table(info, colWidths=[45 * mm, 70 * mm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d2dc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)

    story.append(Spacer(1, 18 * mm))
    story.append(p(
        "Отчёт сформирован автоматически по результатам выполнения скрипта "
        "<b>lab3_solution.py</b>. Все исходные таблицы и графики находятся "
        "в папке <b>results/</b>.",
        CENTER,
    ))
    story.append(PageBreak())

    # ======================= 1. Цель работы =======================
    story.append(p("1. Цель и постановка задачи", H1))
    story.append(p(
        "Цель работы — разработка и оценка регрессионных моделей машинного "
        "обучения для прогнозирования временного ряда энергопотребления. "
        "Используется дневной набор данных OPSD Germany daily за 2006–2017 гг. "
        "Обучение итоговой модели выполняется на 10 годах (2006–2016), "
        "проверка — на 2017 годе."
    ))
    story.append(p("В работе решаются следующие задачи:", BODY))
    story.append(bullet(
        "загрузить и проанализировать набор данных; построить графики ряда, "
        "скользящих средних и автокорреляционную функцию;"
    ))
    story.append(bullet(
        "подготовить календарные и гармонические (Fourier) признаки, "
        "описывающие суточную, недельную и годовую сезонность;"
    ))
    story.append(bullet(
        "<b>Задание 2:</b> оценить точность пяти алгоритмов машинного обучения "
        "при изменении размера тестовой выборки от 1 до 12 месяцев, построить "
        "таблицы метрик R2, MSE, RMSE, MAE и графики их зависимостей;"
    ))
    story.append(bullet(
        "<b>Задание 3:</b> выбрать лучший алгоритм и выполнить прогноз "
        "энергопотребления на 1 год вперёд средствами библиотеки sklearn."
    ))
    story.append(p("Использованные алгоритмы:", BODY))

    models_table = [
        ["Обозначение", "Алгоритм sklearn", "Краткая характеристика"],
        ["LR", "LinearRegression", "Линейная регрессия; базовая модель, хорошо интерпретируется"],
        ["NN", "MLPRegressor", "Многослойный перцептрон (64–32 нейрона), нелинейная модель"],
        ["KNN", "KNeighborsRegressor", "Метод k ближайших соседей (k = 5, взвешивание по расстоянию)"],
        ["RF", "RandomForestRegressor", "Случайный лес из 200 деревьев, устойчив к выбросам"],
        ["SVR", "SVR (RBF)", "Машина опорных векторов с радиальным ядром"],
    ]
    story.append(styled_table(
        models_table,
        col_widths=[24 * mm, 48 * mm, 96 * mm],
        font_size=8.5,
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(p(
        "Для KNN, SVR и нейронной сети признаки масштабируются "
        "(StandardScaler). Для SVR дополнительно масштабируется целевая "
        "переменная, иначе при C = 1 и ε = 0.1 модель недообучается на "
        "значениях порядка 1300 ГВт·ч.",
        SMALL,
    ))

    # ======================= 2. Данные и EDA =======================
    story.append(PageBreak())
    story.append(p("2. Исходные данные и разведочный анализ", H1))
    story.append(p(
        "Набор данных содержит 4383 дневных наблюдения за период "
        "с 01.01.2006 по 31.12.2017. Столбец <b>Consumption</b> — "
        "энергопотребление в ГВт·ч; также присутствуют столбцы Wind, Solar и "
        "Wind+Solar, которые в данной работе не используются. Пропусков в ряде "
        "Consumption нет."
    ))
    story.append(figure(
        "01_consumption_and_trends.png",
        "Рис. 1. Дневное энергопотребление, скользящие средние за 7 и 365 дней.",
    ))
    story.append(p(
        "На графике видны два основных эффекта: недельная цикличность "
        "(будни выше выходных) и годовая сезонность (зимний максимум, летний "
        "минимум). Долгосрочный тренд выражен слабо, поэтому при прогнозе "
        "основное внимание уделяется сезонным признакам."
    ))
    story.append(figure(
        "02_autocorrelation.png",
        "Рис. 2. Автокорреляционная функция дневного потребления (2012–2013).",
    ))
    story.append(p(
        "АКФ подтверждает сильную периодичность: пики повторяются с недельным "
        "шагом, а медленно затухающая огибающая отражает годовую сезонность. "
        "Это обосновывает добавление недельных и годовых гармоник в признаки."
    ))

    # ======================= 3. Методика =======================
    story.append(PageBreak())
    story.append(p("3. Подготовка признаков и методика оценки", H1))
    story.append(p("3.1. Признаки", H2))
    story.append(p(
        "К каждому дню добавляются календарные признаки: день недели, месяц, "
        "квартал, номер дня в году, день месяца, номер недели в году и признак "
        "выходного дня. Дополнительно вводятся гармонические (Fourier) "
        "признаки:"
    ))
    story.append(bullet(
        "годовые гармоники sin/cos с периодами 1, 1/2 и 1/3 года — гладкое "
        "описание сезонности без разрыва между 31 декабря и 1 января;"
    ))
    story.append(bullet(
        "недельная гармоника sin/cos — описание недельного цикла."
    ))
    story.append(p(
        "Лаговые признаки (<b>Yesterday</b>, <b>Yesterday_Diff</b>) из методички "
        "сознательно не используются. Они дают прогноз только на один шаг вперёд, "
        "поскольку для тестового дня берётся реальное потребление за вчера. "
        "Для честного многомесячного прогноза это было бы утечкой данных, "
        "поэтому модель использует только календарь."
    ))

    story.append(p("3.2. Метрики качества", H2))
    story.append(p(
        "Для оценки используются четыре метрики: коэффициент детерминации "
        "<b>R2</b> (чем ближе к 1, тем лучше), средняя квадратичная ошибка "
        "<b>MSE</b>, её корень <b>RMSE</b> (в ГВт·ч) и средняя абсолютная "
        "ошибка <b>MAE</b> (в ГВт·ч)."
    ))

    story.append(p("3.3. Схема задания 2", H2))
    story.append(p(
        "Для каждого горизонта N = 1…12 месяцев тестовая выборка формируется "
        "как последние N календарных месяцев ряда, а обучение выполняется на "
        "всей предшествующей истории. Таким образом, при росте N одновременно "
        "увеличивается горизонт прогноза и сокращается объём доступных "
        "обучающих данных. Все пять моделей обучаются заново для каждого N."
    ))

    # ======================= 4. Задание 2 =======================
    story.append(PageBreak())
    story.append(p("4. Задание 2. Точность при разном размере тестовой выборки", H1))
    story.append(p(
        "На рис. 3 показана зависимость четырёх метрик от горизонта прогноза "
        "для всех пяти алгоритмов. Полные таблицы сохранены в файлах "
        "<b>results/metrics_vs_horizon.csv</b>, "
        "<b>results/metrics_vs_horizon.xlsx</b> и "
        "<b>results/metrics_vs_horizon.md</b>."
    ))
    story.append(figure(
        "05_metrics_vs_horizon.png",
        "Рис. 3. Зависимость R2, RMSE, MAE и MSE от размера тестовой выборки "
        "(горизонта прогноза) для пяти моделей.",
        max_height=120 * mm,
    ))

    story.append(p("Таблица 1. Коэффициент детерминации R2", H2))
    story.append(pivot_to_table(pivots["R2"], "{:.4f}", "Горизонт, мес."))
    story.append(Spacer(1, 3 * mm))

    story.append(p("Таблица 2. Средняя квадратичная ошибка MSE", H2))
    story.append(pivot_to_table(pivots["MSE"], "{:.2f}", "Горизонт, мес."))
    story.append(Spacer(1, 6 * mm))

    story.append(p("Таблица 3. Корень из средней квадратичной ошибки RMSE, ГВт·ч", H2))
    story.append(pivot_to_table(pivots["RMSE"], "{:.2f}", "Горизонт, мес."))
    story.append(PageBreak())

    story.append(p("Таблица 4. Средняя абсолютная ошибка MAE, ГВт·ч", H2))
    story.append(pivot_to_table(pivots["MAE"], "{:.2f}", "Горизонт, мес."))
    story.append(Spacer(1, 5 * mm))

    story.append(p("Основные наблюдения", H2))
    story.append(bullet(
        f"Наилучший результат по R2 на всём диапазоне горизонтов показывает "
        f"<b>KNN</b>: средний R2 = {best_mean_r2:.4f}. Для полного года "
        f"(N = 12) R2 = {pivots['R2'].loc[12, 'KNN']:.4f}."
    ))
    story.append(bullet(
        f"Максимальный R2 = {best_horizon_r2:.4f} достигнут моделью "
        f"<b>{best_horizon_model}</b> при горизонте N = {int(best_horizon)} мес."
    ))
    story.append(bullet(
        "Линейная регрессия <b>LR</b> заметно уступает нелинейным моделям: "
        "R2 растёт с 0.26 при N = 1 до 0.70 при N = 12. Это объясняется тем, "
        "что линейная модель плохо описывает сложную сезонную форму ряда."
    ))
    story.append(bullet(
        "Для KNN, RF и SVR качество слабо зависит от горизонта: модели "
        "устойчиво воспроизводят недельный и годовой цикл. На коротких "
        "горизонтах ошибка выше из-за высокой изменчивости зимнего периода."
    ))
    story.append(bullet(
        "Наименьшие ошибки в абсолютном выражении достигаются на горизонтах "
        "5–7 месяцев (летне-осенний период), где ряд более стабилен."
    ))

    # ======================= 5. Стационарность =======================
    story.append(PageBreak())
    story.append(p("5. Корректировка стационарности (демонстрация из методички)", H1))
    story.append(p(
        "В блокноте рассматривается приём выравнивания амплитуды ряда. "
        "Вычисляется 365-дневный тренд, затем поправка "
        "<b>trend_residuals = max(тренд) − тренд</b> прибавляется к исходному "
        "ряду. Там, где тренд ниже максимума, поправка больше; в точке максимума "
        "она равна нулю. В результате средний уровень ряда становится более "
        "постоянным."
    ))
    story.append(figure(
        "03_stationarity_correction.png",
        "Рис. 4. Исходный и выровненный временные ряды, их тренды и постоянный уровень 1600.",
        max_height=95 * mm,
    ))
    story.append(figure(
        "04_stationarity_yearly_stats.png",
        "Рис. 5. Среднее значение и стандартное отклонение по годам до и после корректировки.",
        max_height=85 * mm,
    ))
    story.append(p(
        "Корректировка уменьшает разброс среднегодовых уровней, то есть "
        "приближает ряд к стационарному. В заданиях 2 и 3, однако, "
        "прогнозируется исходное (реальное) энергопотребление, поскольку "
        "именно оно имеет практический смысл."
    ))

    # ======================= 6. Задание 3 =======================
    story.append(PageBreak())
    story.append(p("6. Задание 3. Итоговый прогноз на 1 год", H1))
    story.append(p(
        f"По среднему R2 лучшей признана модель <b>{best_model}</b> "
        f"(средний R2 = {best_mean_r2:.4f}). Она обучена на данных "
        f"2006–2016 гг. и использована для прогноза всего 2017 года."
    ))

    summary_table = [["Модель", "Средний R2", "Средний RMSE", "Средний MAE", "Средний MSE"]]
    for model, row in summary.iterrows():
        summary_table.append([
            str(model),
            f"{row['mean_R2']:.4f}",
            f"{row['mean_RMSE']:.2f}",
            f"{row['mean_MAE']:.2f}",
            f"{row['mean_MSE']:.2f}",
        ])
    story.append(styled_table(
        summary_table,
        col_widths=[28 * mm, 33 * mm, 38 * mm, 38 * mm, 38 * mm],
        font_size=8.5,
    ))
    story.append(p("Таблица 5. Средние метрики моделей по всем горизонтам 1–12 месяцев.", CAPTION))

    story.append(p("6.1. Метрики итогового прогноза", H2))
    final_table = [
        ["Показатель", "Значение"],
        ["Модель", best_model],
        ["R2", f"{metrics['R2']:.4f}"],
        ["MSE", f"{metrics['MSE']:.2f}"],
        ["RMSE, ГВт·ч", f"{metrics['RMSE']:.2f}"],
        ["MAE, ГВт·ч", f"{metrics['MAE']:.2f}"],
        ["Среднее реальное потребление, ГВт·ч", f"{mean_real:.2f}"],
        ["Относительная ошибка RMSE, %", f"{rel_rmse:.2f}"],
    ]
    story.append(styled_table(
        final_table,
        col_widths=[95 * mm, 55 * mm],
        font_size=9,
    ))
    story.append(p("Таблица 6. Качество прогноза на 2017 год.", CAPTION))
    story.append(p(
        f"Модель объясняет {metrics['R2'] * 100:.1f}% дисперсии потребления. "
        f"Средняя абсолютная ошибка составляет {metrics['MAE']:.1f} ГВт·ч, "
        f"или около {metrics['MAE'] / mean_real * 100:.1f}% от среднего уровня "
        f"потребления. Это хороший результат для прогноза на год вперёд "
        f"без метеорологических данных."
    ))

    story.append(figure(
        "07_final_forecast.png",
        "Рис. 6. Реальное и спрогнозированное энергопотребление в 2017 году.",
        max_height=80 * mm,
    ))
    story.append(figure(
        "08_final_scatter.png",
        "Рис. 7. Диаграмма рассеяния «реальность — прогноз».",
        max_height=85 * mm,
    ))
    story.append(figure(
        "09_final_residuals.png",
        "Рис. 8. Распределение остатков прогноза.",
        max_height=70 * mm,
    ))
    story.append(p(
        "Прогноз хорошо воспроизводит недельную цикличность и общую сезонную "
        "форму, но систематически немного занижает потребление: остатки "
        "(реальность минус прогноз) смещены в положительную область. Это "
        "типичный эффект регрессии к среднему; кроме того, модель не знает "
        "погодных условий, которые определяют аномальные зимние пики. "
        "Явных выбросов немного."
    ))

    story.append(PageBreak())
    story.append(p("6.2. Помесячные средние значения прогноза", H2))
    monthly_table = [["Месяц", "Реальное среднее", "Прогноз (среднее)", "|Ошибка|"]]
    for _, row in monthly.iterrows():
        monthly_table.append([
            row["Month"],
            f"{row['Real']:.1f}",
            f"{row['Prediction']:.1f}",
            f"{row['AbsError']:.1f}",
        ])
    story.append(styled_table(
        monthly_table,
        col_widths=[35 * mm, 45 * mm, 45 * mm, 35 * mm],
        font_size=9,
    ))
    story.append(p(
        "Таблица 7. Средние за месяц реальные и прогнозные значения, ГВт·ч.",
        CAPTION,
    ))

    story.append(figure(
        "11_rf_feature_importance.png",
        "Рис. 9. Важность признаков в модели RandomForest (для интерпретации).",
        max_height=90 * mm,
    ))
    story.append(p(
        "Наибольший вклад вносят недельная гармоника sin_week, годовая "
        "гармоника cos_year_1, признак выходного дня и день недели; далее "
        "идут номер дня в году и остальные годовые гармоники. Это "
        "подтверждает, что модель опирается прежде всего на недельную и "
        "годовую сезонность, что согласуется с результатами разведочного "
        "анализа."
    ))

    # ======================= 7. Выводы =======================
    story.append(PageBreak())
    story.append(p("7. Выводы", H1))
    story.append(p(
        "1. Загружен и проанализирован дневной ряд энергопотребления Германии "
        "за 2006–2017 гг. Показаны недельная и годовая сезонность, а также "
        "слабо выраженный долгосрочный тренд."
    ))
    story.append(p(
        "2. Для прогнозирования подготовлены календарные и гармонические "
        "признаки. Отказ от лаговых признаков позволил получить честный "
        "многомесячный прогноз без утечки данных."
    ))
    story.append(p(
        f"3. По результатам задания 2 лучшей моделью оказалась <b>{best_model}</b> "
        f"со средним R2 = {best_mean_r2:.4f} по горизонтам 1–12 месяцев. "
        f"Максимальное качество достигнуто при N = {int(best_horizon)} мес. "
        f"(R2 = {best_horizon_r2:.4f}, модель {best_horizon_model})."
    ))
    story.append(p(
        f"4. Итоговый прогноз на 2017 год моделью {best_model} имеет "
        f"R2 = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.2f} ГВт·ч, "
        f"MAE = {metrics['MAE']:.2f} ГВт·ч. Относительная ошибка RMSE "
        f"составляет {rel_rmse:.2f}% от среднего потребления."
    ))
    story.append(p(
        "5. Нелинейные модели (KNN, RandomForest, SVR) существенно точнее "
        "линейной регрессии, что объясняется сложной нелинейной формой "
        "сезонности. KNN показывает наилучшее среднее качество, однако "
        "требует хранения всей обучающей выборки и медленнее на больших данных. "
        "RandomForest даёт близкое качество и лучше интерпретируется."
    ))
    story.append(p(
        "6. Ограничение работы — отсутствие метеорологических данных "
        "(температуры, солнечной радиации). Именно они определяют "
        "аномальные всплески потребления, поэтому дальнейшее улучшение "
        "точности целесообразно связывать с добавлением погодных признаков "
        "и лаговых переменных при краткосрочном прогнозировании."
    ))

    story.append(Spacer(1, 8 * mm))
    story.append(p("Состав программных файлов", H2))
    files_table = [
        ["Файл / папка", "Назначение"],
        ["lab3_solution.py", "Полное решение лабораторной работы"],
        ["make_pdf_report.py", "Генератор данного PDF-отчёта"],
        ["requirements.txt", "Список зависимостей Python"],
        ["data/", "Скачанный датасет"],
        ["results/", "Таблицы метрик и графики"],
        ["report_pdf/", "Готовый PDF-отчёт"],
    ]
    story.append(styled_table(
        files_table,
        col_widths=[55 * mm, 115 * mm],
        font_size=9,
    ))

    # ------------------------------------------------------------------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="Лабораторная работа №3. Регрессионный анализ временных рядов",
        author="lab3_solution.py",
        subject="Прогнозирование энергопотребления",
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"[pdf] Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    if not RESULTS_DIR.exists():
        raise SystemExit(
            "Папка results/ не найдена. Сначала запустите lab3_solution.py."
        )
    build_report()
