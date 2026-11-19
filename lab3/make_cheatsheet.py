# -*- coding: utf-8 -*-
"""
Генератор PDF-шпаргалки для сдачи и защиты лабораторной работы №3.

Запуск:
    .venv\\Scripts\\python.exe make_cheatsheet.py

Результат:
    cheatsheet_pdf/Шпаргалка_ЛР3.pdf
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Пути и шрифты
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "cheatsheet_pdf"
OUT_PATH = OUT_DIR / "Шпаргалка_ЛР3.pdf"
FONT_DIR = (
    BASE_DIR / ".venv" / "Lib" / "site-packages" / "matplotlib"
    / "mpl-data" / "fonts" / "ttf"
)

pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(
    TTFont("DejaVu-BoldOblique", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf"))
)
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONT_DIR / "DejaVuSansMono.ttf")))
registerFontFamily(
    "DejaVu", normal="DejaVu", bold="DejaVu-Bold",
    italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique",
)

# ---------------------------------------------------------------------------
# Стили
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "TitleRU", parent=styles["Title"], fontName="DejaVu-Bold",
    fontSize=15, leading=18, alignment=TA_CENTER,
    textColor=colors.HexColor("#23445f"), spaceAfter=1,
)
SUBTITLE = ParagraphStyle(
    "SubtitleRU", parent=styles["Normal"], fontName="DejaVu",
    fontSize=7.5, leading=10, alignment=TA_CENTER,
    textColor=colors.HexColor("#666666"), spaceAfter=4,
)
H1 = ParagraphStyle(
    "H1RU", parent=styles["Heading1"], fontName="DejaVu-Bold",
    fontSize=9.8, leading=12, spaceBefore=5, spaceAfter=3,
    backColor=colors.HexColor("#23445f"), textColor=colors.white,
    borderPadding=2, keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2RU", parent=styles["Heading2"], fontName="DejaVu-Bold",
    fontSize=8.6, leading=10.5, spaceBefore=3, spaceAfter=1.5,
    textColor=colors.HexColor("#2f5d86"), keepWithNext=True,
)
BODY = ParagraphStyle(
    "BodyRU", parent=styles["BodyText"], fontName="DejaVu",
    fontSize=7.6, leading=9.6, alignment=TA_JUSTIFY, spaceAfter=1.5,
)
BULLET = ParagraphStyle(
    "BulletRU", parent=BODY, leftIndent=8, bulletIndent=1,
    spaceAfter=1.2,
)
QA = ParagraphStyle(
    "QARU", parent=BODY, fontSize=7.5, leading=9.3,
    alignment=TA_LEFT, spaceAfter=2.0, leftIndent=0,
)
FORMULA = ParagraphStyle(
    "FormulaRU", parent=BODY, fontName="DejaVu", fontSize=7.8, leading=10,
    alignment=TA_CENTER, backColor=colors.HexColor("#eef3f8"),
    borderPadding=2, spaceBefore=1, spaceAfter=2,
)
CODE = ParagraphStyle(
    "CodeRU", fontName="DejaVuMono", fontSize=6.0, leading=7.0,
    textColor=colors.HexColor("#1f2933"),
    backColor=colors.HexColor("#f2f5f8"),
    borderColor=colors.HexColor("#d5dde5"), borderWidth=0.4,
    borderPadding=2, spaceBefore=1, spaceAfter=2,
)
CAPTION = ParagraphStyle(
    "CaptionRU", parent=BODY, fontName="DejaVu-Oblique",
    fontSize=6.6, leading=8, alignment=TA_CENTER,
    textColor=colors.HexColor("#666666"), spaceAfter=2,
)
KEY = ParagraphStyle(
    "KeyRU", parent=BODY, fontSize=7.4, leading=9.2,
    backColor=colors.HexColor("#fff6e5"),
    borderColor=colors.HexColor("#e0b96a"), borderWidth=0.6,
    borderPadding=3, spaceBefore=2, spaceAfter=3,
)

# ---------------------------------------------------------------------------
# Геометрия страницы: две колонки
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = A4
MARGIN_X = 10 * mm
MARGIN_BOTTOM = 11 * mm
HEADER_H = 13 * mm
GAP = 6 * mm
COL_W = (PAGE_W - 2 * MARGIN_X - GAP) / 2
FRAME_H = PAGE_H - HEADER_H - 4 * mm - MARGIN_BOTTOM


def on_page(canvas, doc):
    canvas.saveState()
    # Шапка
    canvas.setFillColor(colors.HexColor("#23445f"))
    canvas.rect(0, PAGE_H - HEADER_H, PAGE_W, HEADER_H, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("DejaVu-Bold", 9)
    canvas.drawString(
        MARGIN_X, PAGE_H - 8.2 * mm,
        "ШПАРГАЛКА • Лабораторная работа №3: регрессионный анализ временных рядов",
    )
    canvas.setFont("DejaVu", 7.5)
    canvas.drawRightString(
        PAGE_W - MARGIN_X, PAGE_H - 8.2 * mm, f"стр. {doc.page}"
    )
    # Подвал
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.setFont("DejaVu", 6.8)
    canvas.drawString(MARGIN_X, 6.5 * mm, "Ильин А.А., группа УИТСм-11, 06.11.2026")
    canvas.drawRightString(
        PAGE_W - MARGIN_X, 6.5 * mm,
        "lab3_solution.py • sklearn • pandas • matplotlib",
    )
    canvas.restoreState()


def p(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str) -> Paragraph:
    return Paragraph(text, BULLET, bulletText="•")


def qa(question: str, answer: str) -> Paragraph:
    return Paragraph(f"<b>В: {question}</b><br/>О: {answer}", QA)


def section(title: str) -> Paragraph:
    return Paragraph(title, H1)


def sub(title: str) -> Paragraph:
    return Paragraph(title, H2)


def formula(text: str) -> Paragraph:
    return Paragraph(text, FORMULA)


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), CODE)


def table(data, col_widths, font_size=6.8):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23445f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#a9b6c2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef3f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# ---------------------------------------------------------------------------
# Содержание шпаргалки
# ---------------------------------------------------------------------------
def build_story() -> list:
    story: list = []

    # ======================= Титул и ключевое =======================
    story.append(p("ШПАРГАЛКА ЛР №3", TITLE))
    story.append(p(
        "К защите • прогнозирование энергопотребления • OPSD Germany daily • sklearn",
        SUBTITLE,
    ))
    story.append(p(
        "<b>Задача:</b> разработать и оценить регрессионные модели МО для "
        "прогноза дневного энергопотребления. <b>Данные:</b> 4383 наблюдения "
        "за 2006–2017 гг., целевая переменная <b>Consumption</b> (ГВт·ч). "
        "<b>Train:</b> 2006–2016 (4018 дней), <b>test:</b> 2017 (365 дней). "
        "<b>Признаки:</b> календарь + Fourier-гармоники (без лагов — чтобы "
        "не было утечки). <b>Модели:</b> LR, MLP, KNN, RF, SVR. <b>Метрики:</b> "
        "R2, MSE, RMSE, MAE.",
        KEY,
    ))
    story.append(p(
        "<b>Лучший результат по среднему R2 на горизонтах 1–12 мес. — KNN:</b> "
        "средний R2 = <b>0.8636</b>. Итоговый прогноз на 2017 год: "
        "R2 = <b>0.8344</b>, RMSE = <b>66.97</b> ГВт·ч, MAE = <b>55.17</b> ГВт·ч, "
        "относительная RMSE = <b>4.84%</b>.",
        KEY,
    ))

    # ======================= 1. Термины =======================
    story.append(section("1. Базовые понятия"))
    story.append(bullet(
        "<b>Временной ряд</b> — последовательность значений, упорядоченных по "
        "времени. Порядок важен: соседние значения коррелированы, поэтому "
        "перемешивать данные нельзя."
    ))
    story.append(bullet(
        "<b>Тренд</b> — долгосрочная тенденция (рост/падение). "
        "<b>Сезонность</b> — повторяющиеся колебания с фиксированным периодом "
        "(неделя, год). <b>Цикл</b> — нерегулярные колебания большей длины. "
        "<b>Шум</b> — случайная составляющая."
    ))
    story.append(bullet(
        "<b>Стационарность</b> — свойства ряда (мат. ожидание, дисперсия, "
        "автоковариация) не зависят от времени. Многие модели и статистические "
        "критерии предполагают стационарность."
    ))
    story.append(bullet(
        "<b>АКФ (автокорреляционная функция)</b> — корреляция ряда с его "
        "сдвинутой копией. Показывает «память» ряда и периодичность. У нас "
        "пики через 7 дней — недельная сезонность, медленное затухание — "
        "годовая."
    ))
    story.append(bullet(
        "<b>Горизонт прогноза</b> — на сколько шагов (дней/месяцев) вперёд "
        "строится прогноз. <b>Лаг</b> — сдвиг ряда назад (вчерашнее значение). "
        "<b>Скользящее среднее</b> — среднее по окну, сглаживает ряд и "
        "выделяет тренд."
    ))
    story.append(bullet(
        "<b>Остатки</b> — e = y − ŷ. В идеале около нуля, без тренда и "
        "периодичности; по ним проверяют систематическое смещение."
    ))

    # ======================= 2. Метрики =======================
    story.append(section("2. Метрики качества"))
    story.append(formula("R2 = 1 − Σ(y − ŷ)² / Σ(y − ȳ)²"))
    story.append(formula("MSE = (1/n) Σ(y − ŷ)²"))
    story.append(formula("RMSE = √MSE            MAE = (1/n) Σ|y − ŷ|"))
    story.append(bullet(
        "R2: 1 — идеально; 0 — модель как среднее; меньше 0 — хуже среднего. "
        "R2 = 0.8344 означает, что объяснено 83.44% дисперсии."
    ))
    story.append(bullet(
        "RMSE — в единицах целевой переменной (ГВт·ч), сильнее штрафует "
        "крупные ошибки. MAE — средняя абсолютная ошибка, устойчивее к "
        "выбросам. MSE — квадрат ошибки, используется для расчётов."
    ))
    story.append(bullet(
        "Не путать: R2 = 0.83 — это <b>не</b> «83% точности», а 83% "
        "объяснённой дисперсии. Точность в процентах можно оценить как "
        "1 − MAE/ȳ (у нас ≈ 96%)."
    ))

    # ======================= 3. Признаки =======================
    story.append(section("3. Признаки и подготовка данных"))
    story.append(bullet(
        "<b>Календарные признаки:</b> dayofweek, month, quarter, dayofyear, "
        "day, weekofyear, is_weekend. Описывают недельную и годовую сезонность."
    ))
    story.append(bullet(
        "<b>Fourier-признаки:</b> sin/cos(2πk·doy/365.25), k = 1…3; "
        "sin/cos(2π·dow/7). Дают гладкую периодическую сезонность без разрыва "
        "31.12 → 01.01, особенно полезны для LR и SVR."
    ))
    story.append(bullet(
        "<b>Почему нет лагов Yesterday / Yesterday_Diff:</b> для прогноза на "
        "год это утечка данных — в тесте мы подставили бы реальное потребление "
        "за вчера. Лаги корректны для прогноза на 1 шаг вперёд."
    ))
    story.append(bullet(
        "<b>Почему нет year:</b> деревья не экстраполируют за пределы "
        "обучающего диапазона; числовой год может только навредить."
    ))
    story.append(bullet(
        "<b>Масштабирование:</b> StandardScaler для NN, KNN и SVR. Для SVR "
        "дополнительно масштабируется целевая переменная (иначе при C = 1 и "
        "ε = 0.1 модель недообучается). Деревья и LR в масштабировании не "
        "нуждаются."
    ))

    # ======================= 4. Модели =======================
    story.append(section("4. Модели sklearn"))
    story.append(bullet(
        "<b>LR — LinearRegression.</b> Линейная связь. Быстро, "
        "интерпретируемо, но плохо описывает нелинейную сезонность; у нас "
        "самая слабая (средний R2 = 0.6020)."
    ))
    story.append(bullet(
        "<b>NN — MLPRegressor.</b> Нейросеть 64–32, relu, adam, "
        "early_stopping. Гибкая, но требует масштабирования, данных и "
        "подбора архитектуры."
    ))
    story.append(bullet(
        "<b>KNN — KNeighborsRegressor.</b> k = 5, weights = 'distance'. "
        "Локально усредняет похожие дни, хорошо ловит нелинейность. Минусы: "
        "хранит всю выборку, медленный на больших данных, чувствителен к "
        "масштабу."
    ))
    story.append(bullet(
        "<b>RF — RandomForestRegressor.</b> Ансамбль 200 деревьев (бэггинг + "
        "случайные признаки). Устойчив, нелинейный, даёт feature_importances_. "
        "Не экстраполирует, тяжелее по памяти."
    ))
    story.append(bullet(
        "<b>SVR — SVR(kernel='rbf').</b> C = 100, ε = 0.1, γ = 'scale'. Ищет "
        "функцию в ε-трубке с минимальной нормой весов. Хорош на средних "
        "данных, чувствителен к C/γ/ε и масштабу."
    ))

    # ======================= 5. Валидация =======================
    story.append(section("5. Валидация временного ряда"))
    story.append(bullet(
        "<b>TimeSeriesSplit:</b> обучение на прошлых блоках, валидация — на "
        "следующем по времени. Сохраняет хронологию, имитирует реальный "
        "прогноз. У нас n_splits = 5."
    ))
    story.append(formula(
        "tscv = TimeSeriesSplit(n_splits=5)<br/>"
        "cross_val_score(model, X, y, cv=tscv, scoring='r2')"
    ))
    story.append(bullet(
        "<b>Почему не KFold:</b> он перемешивает время, будущее попадает в "
        "обучение → утечка и завышенная оценка качества."
    ))
    story.append(bullet(
        "<b>CV R2 (5 фолдов):</b> RF 0.7175, SVR 0.6964, NN 0.6940, "
        "KNN 0.6743, LR 0.6658. По CV лучший — RF, но по среднему R2 на "
        "горизонтах 1–12 — KNN. Вывод зависит от схемы валидации."
    ))

    # ======================= 6. Задание 2 =======================
    story.append(section("6. Задание 2: точность vs размер теста"))
    story.append(p(
        "<b>Схема:</b> для N = 1…12 месяцев тест — последние N календарных "
        "месяцев ряда, обучение — вся история до начала теста. Модели "
        "обучаются заново для каждого N. Так растёт горизонт и сокращается "
        "объём обучающих данных."
    ))
    story.append(code("""
test_start = pd.Timestamp('2018-01-01') - pd.DateOffset(months=N)
train_end  = test_start - pd.Timedelta(days=1)
train = df.loc[:train_end]
test  = df.loc[test_start:'2017-12-31']
"""))
    story.append(sub("Средние метрики по горизонтам 1–12 мес."))
    story.append(table([
        ["Модель", "R2", "RMSE", "MAE", "MSE"],
        ["KNN", "0.8636", "59.33", "49.68", "3532.38"],
        ["RF", "0.8414", "64.04", "55.50", "4120.07"],
        ["SVR", "0.8278", "66.70", "53.01", "4490.33"],
        ["NN", "0.7529", "79.78", "65.51", "6479.04"],
        ["LR", "0.6020", "100.69", "80.89", "10587.97"],
    ], [16 * mm, 17 * mm, 17 * mm, 17 * mm, 20 * mm]))
    story.append(p("Таблица 1. Средние метрики по всем горизонтам.", CAPTION))

    story.append(sub("R2 по горизонтам (ключевые точки)"))
    story.append(table([
        ["N, мес.", "LR", "NN", "KNN", "RF", "SVR"],
        ["1", "0.261", "0.666", "0.883", "0.842", "0.802"],
        ["6", "0.703", "0.820", "0.885", "0.867", "0.876"],
        ["12", "0.700", "0.782", "0.834", "0.822", "0.822"],
    ], [14 * mm, 15 * mm, 15 * mm, 15 * mm, 15 * mm, 15 * mm]))
    story.append(p("Таблица 2. Фрагмент полной таблицы R2.", CAPTION))
    story.append(bullet(
        "<b>Максимум R2 = 0.8848</b> — KNN при N = 6 мес. "
        "У KNN, RF и SVR качество стабильно (0.82–0.88) на всём диапазоне."
    ))
    story.append(bullet(
        "У LR R2 растёт с 0.26 (N = 1) до 0.70 (N = 12): линейная модель "
        "плохо описывает нелинейную сезонность, но на длинном окне "
        "компенсирует это средним уровнем."
    ))
    story.append(bullet(
        "Короткие зимние горизонты сложнее из-за высокой изменчивости; "
        "лучшие абсолютные ошибки — на горизонтах 5–7 мес. (лето–осень)."
    ))

    # ======================= 7. Задание 3 =======================
    story.append(section("7. Задание 3: итоговый прогноз на 2017"))
    story.append(bullet(
        "Лучшая модель по среднему R2 — <b>KNN</b>. Обучение на 2006–2016, "
        "прогноз на весь 2017 год."
    ))
    story.append(table([
        ["Метрика", "Значение"],
        ["R2", "0.8344"],
        ["MSE", "4484.37"],
        ["RMSE, ГВт·ч", "66.97"],
        ["MAE, ГВт·ч", "55.17"],
        ["Среднее реальное, ГВт·ч", "1382.84"],
        ["Относительная RMSE", "4.84%"],
        ["Относительная MAE", "≈ 3.99%"],
    ], [35 * mm, 30 * mm], font_size=7.0))
    story.append(p("Таблица 3. Качество прогноза на 2017 год.", CAPTION))
    story.append(bullet(
        "Прогноз хорошо повторяет недельный цикл и сезонную форму, но "
        "занижает пики: остатки смещены в положительную область (реальность "
        "чаще выше прогноза). Причины — регрессия к среднему и отсутствие "
        "погодных признаков."
    ))
    story.append(bullet(
        "Худший месяц — январь (реальное среднее 1530.4 против прогноза "
        "1406.4, ошибка 123.9). Лучший — март (ошибка 3.8 ГВт·ч)."
    ))

    # ======================= 8. Стационарность =======================
    story.append(section("8. Корректировка стационарности"))
    story.append(p(
        "Приём из методички: выравнивание амплитуды ряда по 365-дневному "
        "тренду."
    ))
    story.append(code("""
trend365 = data['Consumption'].rolling(365, center=True).mean()
trend365 = trend365.bfill().ffill()
resid = trend365.max() - trend365
consumption_eq = data['Consumption'] + resid
trend_eq = trend365 + resid * 2
"""))
    story.append(bullet(
        "Смысл: где тренд ниже максимума — поправка больше; в максимуме — "
        "ноль. Средний уровень становится более постоянным, ряд ближе к "
        "стационарному."
    ))
    story.append(bullet(
        "В заданиях 2 и 3 прогнозируется исходное (реальное) потребление, "
        "поскольку именно оно имеет практический смысл."
    ))

    # ======================= 9. Код =======================
    story.append(section("9. Мини-конспект кода"))
    story.append(code("""
# 1. Данные
data = pd.read_csv(url, parse_dates=['Date']).set_index('Date')
data['Consumption'].rolling(365, center=True).mean()

# 2. Признаки (фрагмент)
df['dayofweek'] = df.index.dayofweek
df['dayofyear'] = df.index.dayofyear
doy = df.index.dayofyear.to_numpy()
df['sin_year_1'] = np.sin(2*np.pi*doy/365.25)
df['cos_year_1'] = np.cos(2*np.pi*doy/365.25)

# 3. Train / test
X_train = df.loc[:'2016-12-31'].drop(columns='Consumption')
y_train = df.loc[:'2016-12-31', 'Consumption']
X_test  = df.loc['2017-01-01':'2017-12-31'].drop(columns='Consumption')
y_test  = df.loc['2017-01-01':'2017-12-31', 'Consumption']

# 4. Модель и прогноз
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Метрики
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error)
r2   = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae  = mean_absolute_error(y_test, y_pred)

# 6. Валидация
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv,
                         scoring='r2')
"""))

    # ======================= 10. Вопросы =======================
    story.append(section("10. Вопросы на защите"))
    story.append(qa(
        "Что такое временной ряд?",
        "Последовательность значений, упорядоченная по времени. Порядок "
        "важен, соседние значения коррелированы."
    ))
    story.append(qa(
        "Основные компоненты ряда?",
        "Тренд, сезонность, циклическая составляющая и случайный шум."
    ))
    story.append(qa(
        "Что такое стационарность и зачем она?",
        "Свойства ряда не зависят от времени (среднее, дисперсия, "
        "автоковариация). Упрощает моделирование и делает корректными "
        "статистические выводы."
    ))
    story.append(qa(
        "Что такое АКФ?",
        "Корреляция ряда со своей сдвинутой копией. Показывает память и "
        "периодичность. Пики через 7 дней — недельная сезонность."
    ))
    story.append(qa(
        "Почему нельзя KFold?",
        "Он перемешивает время: будущее попадает в обучение, возникает "
        "утечка и завышенная оценка. Нужен TimeSeriesSplit."
    ))
    story.append(qa(
        "Что такое TimeSeriesSplit?",
        "Схема кросс-валидации, где обучение идёт на прошлых блоках, а "
        "валидация — на следующем по времени."
    ))
    story.append(qa(
        "Почему выбраны календарные и Fourier признаки?",
        "Они описывают недельную и годовую сезонность. Fourier гладкий и "
        "периодичный, без разрыва на границе года; особенно полезен для "
        "LR и SVR."
    ))
    story.append(qa(
        "Почему не используете лаги Yesterday?",
        "Для многомесячного прогноза это утечка: в тесте мы взяли бы "
        "реальное потребление за вчера. Лаги корректны для one-step-ahead."
    ))
    story.append(qa(
        "Зачем StandardScaler?",
        "KNN, SVR и MLP чувствительны к масштабу признаков. Деревья и LR "
        "в масштабировании не нуждаются. Для SVR масштабируется и y."
    ))
    story.append(qa(
        "Как считается R2 и что значит 0.83?",
        "R2 = 1 − SS_res/SS_tot. 1 — идеально, 0 — как среднее, меньше 0 — "
        "хуже среднего. R2 = 0.83 — объяснено 83% дисперсии."
    ))
    story.append(qa(
        "RMSE, MAE, MSE — в чём разница?",
        "MSE — квадрат ошибки; RMSE — корень из MSE, в единицах y; MAE — "
        "средний модуль. RMSE сильнее штрафует крупные ошибки, MAE "
        "устойчивее к выбросам."
    ))
    story.append(qa(
        "Почему R2 может быть отрицательным?",
        "Модель оказалась хуже константного прогноза среднего на этом тесте."
    ))
    story.append(qa(
        "Как обучали и тестировали?",
        "2006–2016 — обучение, 2017 — тест. В задании 2 тест — последние "
        "N месяцев, обучение — всё до начала теста."
    ))
    story.append(qa(
        "Как точность зависит от горизонта?",
        "У KNN, RF и SVR стабильна (0.82–0.88); у LR растёт с 0.26 до 0.70. "
        "Короткие зимние горизонты сложнее. Максимум R2 = 0.8848 у KNN "
        "при N = 6."
    ))
    story.append(qa(
        "Какая модель лучшая?",
        "По среднему R2 на горизонтах 1–12 — KNN (0.8636). По "
        "TimeSeriesSplit CV — RF (0.7175). Выбор зависит от схемы валидации "
        "и целевого периода."
    ))
    story.append(qa(
        "Почему KNN лучший?",
        "Локально усредняет похожие календарные дни и хорошо воспроизводит "
        "нелинейный сезонный профиль. Минусы: хранит всю выборку, медленный "
        "на больших данных, чувствителен к масштабу."
    ))
    story.append(qa(
        "Почему прогноз занижает пики?",
        "KNN и деревья усредняют соседей (регрессия к среднему), а "
        "температуры и других погодных факторов в признаках нет."
    ))
    story.append(qa(
        "Как улучшить результат?",
        "Добавить погодные признаки (температура, радиация), лаги для "
        "краткосрочного прогноза, подобрать гиперпараметры, попробовать "
        "GradientBoosting/HistGB/LSTM, rolling-origin CV и ансамбли."
    ))
    story.append(qa(
        "Что такое переобучение?",
        "Модель запоминает train и плохо обобщает. Признаки: высокий R2 на "
        "train, низкий на CV/test. Борьба: упрощение, регуляризация, больше "
        "данных, early_stopping, CV."
    ))
    story.append(qa(
        "Что такое data leakage?",
        "Использование при обучении информации, недоступной в момент "
        "прогноза (будущие значения, лаги из теста). Даёт завышенную оценку."
    ))
    story.append(qa(
        "Как работает RandomForest?",
        "Ансамбль деревьев на бутстрэп-выборках и случайных подмножествах "
        "признаков. Усредняет прогнозы, снижает дисперсию, даёт важности "
        "признаков."
    ))
    story.append(qa(
        "Как работает SVR?",
        "Ищет функцию, отклоняющуюся от y не более чем на ε, с минимальной "
        "нормой весов. C — штраф за выход за трубку, γ — ширина RBF-ядра."
    ))
    story.append(qa(
        "Как работает KNN?",
        "Для объекта находит k ближайших соседей и усредняет их y; у нас "
        "k = 5 и weights = 'distance'."
    ))
    story.append(qa(
        "Что такое MLP?",
        "Многослойный перцептрон: вход → скрытые слои с нелинейностью → "
        "выход. Обучается backpropagation; у нас adam + early_stopping."
    ))
    story.append(qa(
        "Что такое остатки и как их проверять?",
        "e = y − ŷ. В идеале около нуля, без тренда и периодичности. У нас "
        "есть положительное смещение — модель немного занижает потребление."
    ))
    story.append(qa(
        "Что такое скользящее среднее?",
        "Среднее по окну: rolling(7/365, center=True). Сглаживает ряд и "
        "выделяет тренд; center=True центрирует окно."
    ))
    story.append(qa(
        "Что делают shift() и diff()?",
        "shift() сдвигает ряд (лаг), diff() — разность соседних значений "
        "(прирост). Используются для лаговых признаков."
    ))
    story.append(qa(
        "Что такое горизонт прогноза?",
        "Количество шагов (дней или месяцев) вперёд, на которое строится "
        "прогноз."
    ))
    story.append(qa(
        "Почему train 10 лет, test 1 год?",
        "10 лет дают много полных сезонных циклов для обучения, а год — "
        "проверку на всех сезонах."
    ))
    story.append(qa(
        "Чем регрессия отличается от классификации?",
        "В регрессии целевая переменная непрерывная (ГВт·ч), в классификации — "
        "дискретная (класс). У нас регрессия."
    ))
    story.append(qa(
        "Что такое naive (baseline) прогноз?",
        "Простейший ориентир: «завтра = сегодня» или «сегодня = тот же день "
        "год назад». Если модель не лучше naive, она бесполезна."
    ))
    story.append(qa(
        "Почему MAE меньше RMSE?",
        "RMSE возводит ошибки в квадрат и сильнее штрафует крупные промахи. "
        "У нас MAE = 55.17, RMSE = 66.97 ГВт·ч."
    ))
    story.append(qa(
        "Что такое медианная абсолютная ошибка?",
        "Медиана |y − ŷ|. Более устойчива к выбросам, чем MAE. В методичке "
        "она также упоминается."
    ))
    story.append(qa(
        "Что такое гиперпараметры и как их подбирать?",
        "Параметры, задаваемые до обучения (k в KNN, число деревьев, C/γ в "
        "SVR). Подбирают GridSearchCV/RandomizedSearchCV с TimeSeriesSplit."
    ))
    story.append(qa(
        "Что такое Pipeline и зачем он?",
        "Цепочка «преобразование → модель». Гарантирует, что scaler "
        "обучается только на train, и упрощает кросс-валидацию."
    ))
    story.append(qa(
        "Что такое TransformedTargetRegressor?",
        "Обёртка, которая масштабирует y перед обучением и делает обратное "
        "преобразование прогноза. Используется для SVR."
    ))
    story.append(qa(
        "Что такое feature_importances_?",
        "Важность признаков в деревьях: насколько часто и сильно признак "
        "уменьшает ошибку. У нас лидируют sin_week и cos_year_1."
    ))
    story.append(qa(
        "Что такое GradientBoosting?",
        "Ансамбль, где каждое следующее дерево исправляет ошибки предыдущих. "
        "Часто точнее RF, но требует аккуратной настройки."
    ))
    story.append(qa(
        "Что такое бутстрэп и ансамбль?",
        "Бутстрэп — выборка с возвращением. Ансамбль — объединение моделей "
        "(например, усреднение деревьев в RandomForest) для снижения "
        "дисперсии."
    ))
    story.append(qa(
        "Как сохранить обученную модель?",
        "joblib.dump(model, 'model.pkl') и joblib.load('model.pkl'). Для "
        "больших моделей удобно хранить pipeline целиком."
    ))
    story.append(qa(
        "Что такое rolling-origin валидация?",
        "Последовательно сдвигаем момент прогноза вперёд и каждый раз "
        "обучаемся только на прошлом. Более честная имитация реальной "
        "эксплуатации, чем один фиксированный split."
    ))

    # ======================= 11. Типичные ошибки =======================
    story.append(section("11. Типичные ошибки на защите"))
    story.append(bullet("Использовать KFold для временного ряда — нельзя."))
    story.append(bullet(
        "Брать лаги из тестовой выборки — это утечка данных."
    ))
    story.append(bullet(
        "Забыть StandardScaler для KNN/SVR/MLP — качество резко падает."
    ))
    story.append(bullet("Путать MSE и RMSE; RMSE — в ГВт·ч, MSE — в ГВт·ч²."))
    story.append(bullet(
        "Говорить «R2 = 0.83 — это 83% точности». Правильно: 83% "
        "объяснённой дисперсии."
    ))
    story.append(bullet(
        "Сравнивать модели по качеству на train, а не на отложенном тесте."
    ))
    story.append(bullet(
        "Забыть, что R2 может быть отрицательным."
    ))

    # ======================= 12. План ответа =======================
    story.append(section("12. План ответа на защите"))
    story.append(bullet(
        "<b>1. Постановка.</b> Временной ряд энергопотребления, 2006–2017, "
        "цель — прогноз на год; train 2006–2016, test 2017."
    ))
    story.append(bullet(
        "<b>2. Подготовка.</b> Календарные и Fourier-признаки, без лагов "
        "(честный многомесячный прогноз), масштабирование для KNN/SVR/NN."
    ))
    story.append(bullet(
        "<b>3. Модели.</b> LR, MLP, KNN, RF, SVR; сравнение по R2, MSE, "
        "RMSE, MAE."
    ))
    story.append(bullet(
        "<b>4. Валидация.</b> TimeSeriesSplit вместо KFold; отдельно "
        "исследован горизонт 1–12 месяцев."
    ))
    story.append(bullet(
        "<b>5. Результаты.</b> По горизонтам лучший — KNN (средний R2 = "
        "0.8636, максимум 0.8848 при N = 6). Прогноз на 2017: R2 = 0.8344, "
        "RMSE = 66.97, MAE = 55.17 ГВт·ч."
    ))
    story.append(bullet(
        "<b>6. Выводы и ограничения.</b> Модель хорошо ловит сезонность, но "
        "занижает пики; нет погодных данных. Улучшения: температура, лаги "
        "для краткосрочного прогноза, подбор гиперпараметров, бустинг."
    ))

    # ======================= 13. Шпаргалка-числа =======================
    story.append(section("13. Ключевые числа"))
    story.append(table([
        ["Параметр", "Значение"],
        ["Период данных", "2006–2017"],
        ["Наблюдений", "4383"],
        ["Train / test", "4018 / 365 дней"],
        ["Моделей", "5"],
        ["Горизонтов в задании 2", "12"],
        ["Средний R2 (KNN)", "0.8636"],
        ["Максимальный R2", "0.8848 (KNN, N=6)"],
        ["R2 на 2017 (KNN)", "0.8344"],
        ["RMSE на 2017", "66.97 ГВт·ч"],
        ["MAE на 2017", "55.17 ГВт·ч"],
        ["Среднее потребление", "1382.84 ГВт·ч"],
        ["Относительная RMSE", "4.84%"],
    ], [45 * mm, 42 * mm], font_size=6.9))

    return story


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_left = Frame(
        MARGIN_X, MARGIN_BOTTOM, COL_W, FRAME_H,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="left",
    )
    frame_right = Frame(
        MARGIN_X + COL_W + GAP, MARGIN_BOTTOM, COL_W, FRAME_H,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="right",
    )
    template = PageTemplate(
        id="two_columns", frames=[frame_left, frame_right], onPage=on_page
    )

    doc = BaseDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=HEADER_H,
        bottomMargin=MARGIN_BOTTOM,
        title="Шпаргалка к защите ЛР №3",
        author="Ильин А.А., УИТСм-11",
        subject="Регрессионный анализ временных рядов",
    )
    doc.addPageTemplates([template])
    doc.build(build_story())
    print(f"[pdf] Шпаргалка сохранена: {OUT_PATH}")


if __name__ == "__main__":
    build()
