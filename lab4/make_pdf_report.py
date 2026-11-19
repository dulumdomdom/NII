# -*- coding: utf-8 -*-
"""
Генератор PDF-отчёта по лабораторной работе №4.

Читает таблицы и графики из results/ (их создаёт lab4_solution.py) и собирает
отчёт report_pdf/Лабораторная_работа_4.pdf.

Запуск:
    .venv\\Scripts\\python.exe make_pdf_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus.flowables import HRFlowable

from report_analysis import ANALYSIS_T1, ANALYSIS_T2, ANALYSIS_T3, CONCLUSIONS
from pdf_style import (CAPTION, CENTER, H1, H2, SMALL, SUBTITLE, TITLE, BODY, bullet,
                       code, figure, formula, num, p, table)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parent
RES = Path(__import__("os").environ.get("LAB4_RESULTS", BASE / "results"))
OUT_DIR = BASE / "report_pdf"
OUT = OUT_DIR / "Лабораторная_работа_4.pdf"


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(18 * mm, 10 * mm, "Лабораторная работа №4. Введение в нейронные сети")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"стр. {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#c8d2dc"))
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.restoreState()


def load():
    r = {}
    r["neuron"] = pd.read_csv(RES / "00_single_neuron.csv", encoding="utf-8-sig")
    r["base"] = pd.read_csv(RES / "02_baseline.csv", encoding="utf-8-sig").set_index("optimizer")
    r["last10"] = pd.read_csv(RES / "03_last10_predictions.csv")
    r["softmax0"] = np.loadtxt(RES / "03_first_softmax_vector.txt")
    r["cr"] = pd.read_csv(RES / "04_classification_report.csv", index_col=0, encoding="utf-8-sig")
    r["t1"] = pd.read_csv(RES / "task1_layers.csv", encoding="utf-8-sig")
    r["t1raw"] = pd.read_csv(RES / "task1_layers_raw.csv", encoding="utf-8-sig")
    r["t2"] = pd.read_csv(RES / "task2_test_size.csv", encoding="utf-8-sig")
    r["t3"] = pd.read_csv(RES / "task3_optimizers.csv", encoding="utf-8-sig")
    r["curves"] = pd.read_csv(RES / "task3_curves.csv", encoding="utf-8-sig")
    r["seeds"] = sorted(r["t1raw"].seed.unique())
    return r


def build():
    r = load()
    base, t1, t2, t3 = r["base"], r["t1"], r["t2"], r["t3"]
    a1 = t1[t1.optimizer == "adam"].set_index("hidden_layers")
    g1 = t1[t1.optimizer == "sgd"].set_index("hidden_layers")
    t3i = t3.set_index("optimizer")
    n_seeds = len(r["seeds"])
    s = []

    # ---------------- Титул ----------------
    s.append(Spacer(1, 28 * mm))
    s.append(p("ЛАБОРАТОРНАЯ РАБОТА №4", TITLE))
    s.append(Spacer(1, 4 * mm))
    s.append(p("Введение в нейронные сети", SUBTITLE))
    s.append(Spacer(1, 3 * mm))
    s.append(p("Классификация рукописных цифр MNIST полносвязной нейронной сетью (Keras)", SUBTITLE))
    s.append(Spacer(1, 10 * mm))
    s.append(HRFlowable(width="60%", thickness=1.2, color=colors.HexColor("#23445f"), hAlign="CENTER"))
    s.append(Spacer(1, 10 * mm))
    info = Table([["Выполнил:", "Ильин А.А."], ["Группа:", "УИТСм-11"], ["Дата:", "19.11.2026"]],
                 colWidths=[45 * mm, 70 * mm])
    info.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d2dc")),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    s.append(info)
    s.append(Spacer(1, 18 * mm))
    s.append(p("Отчёт сформирован по результатам выполнения скрипта <b>lab4_solution.py</b>. "
               "Исходные таблицы и графики находятся в папке <b>results/</b>.", CENTER))
    s.append(PageBreak())

    # ---------------- 1. Цель ----------------
    s.append(p("1. Цель и задачи работы", H1))
    s.append(p("<b>Цель:</b> изучить принципы построения и обучения искусственной нейронной "
               "сети на примере классификации рукописных цифр из набора данных MNIST."))
    s.append(p("<b>Задачи:</b>"))
    s.append(num("Исследовать зависимость точности классификации от количества скрытых слоёв "
                 "(от 1 до 5) при фиксированном числе нейронов в слое и прочих параметрах.", 1))
    s.append(num("Оценить влияние размера тестовой выборки (от 5% до 50% с шагом 5%) на "
                 "точность классификации.", 2))
    s.append(num("Исследовать влияние алгоритма оптимизации на качество обучения сети.", 3))
    s.append(p("<b>Инструменты:</b> Python 3.10, TensorFlow 2.19 / Keras 3, scikit-learn 1.7 "
               "(загрузка данных, разбиение, метрики), numpy, pandas, matplotlib, seaborn."))

    # ---------------- 2. Теория ----------------
    s.append(p("2. Краткие теоретические сведения", H1))
    s.append(p("<b>Искусственный нейрон</b> вычисляет взвешенную сумму входов со смещением и "
               "пропускает её через нелинейную функцию активации:"))
    s.append(formula("r = Σ w<sub>i</sub>x<sub>i</sub> + b,     y = σ(r) = 1 / (1 + e<super>−r</super>)"))
    s.append(p("В методичке это показано на примере определения пола по весу и росту с "
               "вручную заданными весами w₁ = 0.3, w₂ = 0.1, b = −39:"))
    nr = r["neuron"]
    rows = [list(nr.columns)] + [[str(v) for v in row] for row in nr.values]
    s.append(table(rows, col_widths=[22 * mm, 16 * mm, 16 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm]))
    s.append(p("Таблица 1. Работа одного нейрона (все три объекта классифицированы верно).", CAPTION))
    s.append(p("<b>Многослойный перцептрон</b> состоит из полносвязных (Dense) слоёв таких "
               "нейронов. Выходной слой из 10 нейронов с функцией <b>softmax</b> выдаёт "
               "вероятности классов 0–9. Сеть обучается минимизацией функции потерь "
               "<b>categorical cross-entropy</b> L = −log p<sub>верного класса</sub> методом "
               "обратного распространения ошибки: градиенты вычисляются по правилу цепочки от "
               "выхода к входу, а <b>оптимизатор</b> обновляет веса: w ← w − η·g (SGD) или по "
               "более сложному адаптивному правилу (Adam, RMSprop и др.). Обучение идёт "
               "мини-батчами (32 примера), один проход по всей выборке — эпоха."))

    # ---------------- 3. Данные ----------------
    s.append(p("3. Данные и предобработка", H1))
    s.append(p("Набор MNIST загружается функцией <font face='DejaVuMono'>fetch_openml('mnist_784')</font>: "
               "70 000 изображений 28×28 пикселей, развёрнутых в векторы длины 784, яркость "
               "0…255; метки — цифры 0–9 (классы сбалансированы, ≈7 000 на класс)."))
    s.append(figure(RES / "01_mnist_examples.png", "Рис. 1. Примеры изображений MNIST с метками.",
                    max_height=30 * mm))
    s.append(bullet("Пиксели нормируются в диапазон [−1, 1]: X' = 2·(X − min)/ptp − 1, "
                    "где ptp = max − min = 255 (одинаково для train и test)."))
    s.append(bullet("Метки кодируются one-hot (to_categorical): 3 → [0,0,0,1,0,0,0,0,0,0]."))
    s.append(bullet("Разбиение train_test_split со stratify=y (доли цифр одинаковы в train и "
                    "test); в базовой модели test_size = 0.25 → 52 500 / 17 500."))

    # ---------------- 4. Базовая модель ----------------
    s.append(p("4. Базовая модель из методички", H1))
    s.append(p("Архитектура: вход 784 → Dense(64, sigmoid) → Dense(64, sigmoid) → "
               "Dense(10, softmax); 50 240 + 4 160 + 650 = <b>55 050</b> обучаемых параметров; "
               "loss = categorical_crossentropy, 10 эпох, batch_size = 32."))
    s.append(code("""
model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, to_categorical(y_train), epochs=10, batch_size=32)
model.evaluate(X_test, to_categorical(y_test))
"""))
    s.append(table([
        ["Оптимизатор", "Test accuracy", "Test loss", "Train accuracy", "Время обучения, с"],
        ["SGD (lr = 0.01)", f"{base.loc['sgd', 'test_acc']:.4f}", f"{base.loc['sgd', 'test_loss']:.4f}",
         f"{base.loc['sgd', 'train_acc']:.4f}", f"{base.loc['sgd', 'fit_seconds']:.0f}"],
        ["Adam (lr = 0.001)", f"{base.loc['adam', 'test_acc']:.4f}", f"{base.loc['adam', 'test_loss']:.4f}",
         f"{base.loc['adam', 'train_acc']:.4f}", f"{base.loc['adam', 'fit_seconds']:.0f}"],
    ], col_widths=[38 * mm, 32 * mm, 30 * mm, 32 * mm, 36 * mm]))
    s.append(p("Таблица 2. Качество базовой модели (каждая модель обучена с нуля).", CAPTION))
    s.append(p("В методичке после обучения с SGD вызывается повторный compile с Adam. Повторный "
               "compile не сбрасывает веса, поэтому там Adam дообучает уже обученную сеть. В "
               "нашем решении обе модели обучаются с нуля — сравнение оптимизаторов честное.",
               SMALL))
    l10 = r["last10"]
    s.append(p(f"Прогноз для 10 последних изображений тестовой выборки (argmax по выходу "
               f"softmax): <b>{' '.join(map(str, l10.pred))}</b>; истинные метки: "
               f"<b>{' '.join(map(str, l10.true))}</b> — совпало {int((l10.pred == l10.true).sum())} из 10. "
               f"Для первого из них сеть выдала вероятность {l10.p_max.iloc[0]:.4f} для цифры "
               f"{l10.pred.iloc[0]}."))
    s.append(figure(RES / "04_confusion_matrix.png", "Рис. 2. Матрица ошибок базовой модели (Adam).",
                    max_height=85 * mm))
    cr = r["cr"]
    digits = [str(i) for i in range(10)]
    crd = cr.loc[digits]
    worst = crd["f1-score"].astype(float).idxmin()
    best = crd["f1-score"].astype(float).idxmax()
    s.append(p(f"Лучше всего распознаётся цифра <b>{best}</b> (F1 = {float(crd.loc[best, 'f1-score']):.4f}), "
               f"хуже всего — <b>{worst}</b> (F1 = {float(crd.loc[worst, 'f1-score']):.4f}). Типичные "
               "ошибки — похожие по начертанию пары (4↔9, 3↔5, 7↔9, 2↔7), см. рис. 3."))
    s.append(figure(RES / "05_errors.png", "Рис. 3. Примеры ошибочно классифицированных изображений.",
                    max_height=65 * mm))

    # ---------------- Методика ----------------
    s.append(p("5. Методика экспериментов", H1))
    s.append(p("Во всех экспериментах меняется только исследуемый фактор, прочие параметры "
               "фиксированы:"))
    s.append(table([
        ["Параметр", "Значение"],
        ["Нейронов в скрытом слое", "64"],
        ["Функция активации скрытых слоёв", "sigmoid (как в методичке)"],
        ["Выходной слой", "Dense(10, softmax)"],
        ["Функция потерь", "categorical_crossentropy"],
        ["Число эпох / размер батча", "10 / 32"],
        ["Число скрытых слоёв (задания 2, 3)", "2"],
        ["Доля тестовой выборки (задания 1, 3)", "25%"],
        ["Оптимизатор (задания 1, 2)", "Adam (в задании 1 дополнительно SGD)"],
        ["Повторы", f"{n_seeds} запуска с random seed = {', '.join(map(str, r['seeds']))}"],
    ], col_widths=[80 * mm, 90 * mm]))
    s.append(p("Каждый запуск с новым seed меняет и разбиение данных, и начальные веса. В "
               "таблицах приводится среднее и стандартное отклонение по повторам: это "
               "позволяет отличить реальный эффект от случайного разброса.", SMALL))

    # ---------------- 6. Задание 1 ----------------
    s.append(p("6. Задание 1. Зависимость точности от числа скрытых слоёв", H1))
    s.append(figure(RES / "06_task1_layers.png",
                    "Рис. 4. Точность на тесте (среднее ± std) и на обучении в зависимости от "
                    "числа скрытых слоёв для оптимизаторов Adam и SGD.", max_height=68 * mm))
    rows = [["Скрытых слоёв", "Параметров", "Adam: test", "Adam: train", "SGD: test", "SGD: train"]]
    for n in range(1, 6):
        rows.append([str(n), f"{int(a1.loc[n, 'params']):,}".replace(",", " "),
                     f"{a1.loc[n, 'test_acc_mean']:.4f} ± {a1.loc[n, 'test_acc_std']:.4f}",
                     f"{a1.loc[n, 'train_acc_mean']:.4f}",
                     f"{g1.loc[n, 'test_acc_mean']:.4f} ± {g1.loc[n, 'test_acc_std']:.4f}",
                     f"{g1.loc[n, 'train_acc_mean']:.4f}"])
    s.append(table(rows, col_widths=[24 * mm, 22 * mm, 36 * mm, 24 * mm, 36 * mm, 24 * mm], font_size=8))
    s.append(p("Таблица 3. Accuracy в зависимости от числа скрытых слоёв (64 нейрона в слое).", CAPTION))
    for t in ANALYSIS_T1(a1, g1):
        s.append(bullet(t))

    # ---------------- 7. Задание 2 ----------------
    s.append(p("7. Задание 2. Влияние размера тестовой выборки", H1))
    s.append(figure(RES / "07_task2_test_size.png",
                    "Рис. 5. Точность в зависимости от доли тестовой выборки (столбцы — размер "
                    "обучающей выборки).", max_height=72 * mm))
    rows = [["Доля теста", "Train, шт.", "Test, шт.", "Test accuracy", "Train accuracy", "95% интервал оценки"]]
    for _, row in t2.iterrows():
        se = 1.96 * np.sqrt(row.test_acc_mean * (1 - row.test_acc_mean) / row.n_test)
        rows.append([f"{row.test_size:.0%}", f"{int(row.n_train)}", f"{int(row.n_test)}",
                     f"{row.test_acc_mean:.4f} ± {row.test_acc_std:.4f}",
                     f"{row.train_acc_mean:.4f}", f"± {se:.4f}"])
    s.append(table(rows, col_widths=[22 * mm, 24 * mm, 22 * mm, 38 * mm, 28 * mm, 34 * mm], font_size=8))
    s.append(p("Таблица 4. Accuracy при разной доле тестовой выборки (2 скрытых слоя, Adam). "
               "Последний столбец — 95%-я статистическая погрешность accuracy, оценённой на "
               "тесте такого размера: 1.96·√(p(1−p)/n).", CAPTION))
    for t in ANALYSIS_T2(t2):
        s.append(bullet(t))

    # ---------------- 8. Задание 3 ----------------
    s.append(p("8. Задание 3. Влияние алгоритма оптимизации", H1))
    s.append(p("Сравнивались 8 оптимизаторов Keras со стандартными скоростями обучения: "
               "SGD (η = 0.01), SGD с импульсом 0.9 (η = 0.01), Adagrad, Adadelta, RMSprop, Adam, "
               "Nadam, Adamax (η = 0.001). Архитектура — базовая (2 × 64, sigmoid), 10 эпох."))
    s.append(table([
        ["Оптимизатор", "Правило обновления (упрощённо)"],
        ["SGD", "w ← w − η·g"],
        ["SGD + momentum", "v ← 0.9·v − η·g;  w ← w + v"],
        ["Adagrad", "G ← G + g²;  w ← w − η·g/√G  (шаг постоянно уменьшается)"],
        ["Adadelta", "шаг = √(ср. квадрат прошлых шагов) / √(ср. квадрат градиентов) · g"],
        ["RMSprop", "E ← ρ·E + (1−ρ)·g²;  w ← w − η·g/√E"],
        ["Adam", "импульс m + RMSprop-нормировка v + поправка смещения"],
        ["Nadam", "Adam с импульсом Нестерова"],
        ["Adamax", "Adam с нормой L∞ вместо L2"],
    ], col_widths=[38 * mm, 132 * mm], font_size=8))
    s.append(Spacer(1, 3 * mm))
    s.append(figure(RES / "08_task3_curves.png",
                    "Рис. 6. Кривые обучения: функция потерь на обучающей выборке (лог. шкала) "
                    "и точность на тестовой выборке после каждой эпохи.", max_height=68 * mm))
    s.append(figure(RES / "09_task3_bars.png", "Рис. 7. Итоговая точность после 10 эпох.",
                    max_height=62 * mm))
    rows = [["Оптимизатор", "Test accuracy", "Test loss", "Train accuracy", "Время, с"]]
    for _, row in t3.iterrows():
        rows.append([row.optimizer, f"{row.test_acc_mean:.4f} ± {row.test_acc_std:.4f}",
                     f"{row.test_loss_mean:.4f}", f"{row.train_acc_mean:.4f}",
                     f"{row.fit_seconds_mean:.0f}"])
    s.append(table(rows, col_widths=[34 * mm, 40 * mm, 30 * mm, 34 * mm, 26 * mm], font_size=8))
    s.append(p("Таблица 5. Результаты сравнения оптимизаторов (отсортировано по точности).", CAPTION))
    for t in ANALYSIS_T3(t3i, r["curves"]):
        s.append(bullet(t))

    # ---------------- 9. Выводы ----------------
    s.append(PageBreak())
    s.append(p("9. Выводы", H1))
    for i, t in enumerate(CONCLUSIONS(base, a1, g1, t2, t3i), 1):
        s.append(num(t, i))

    s.append(p("Состав программных файлов", H2))
    s.append(table([
        ["Файл / папка", "Назначение"],
        ["lab4_solution.py", "Полное решение: базовая модель и задания 1–3"],
        ["make_pdf_report.py, report_analysis.py, pdf_style.py", "Генератор данного отчёта"],
        ["requirements.txt", "Зависимости Python"],
        ["results/", "Таблицы (CSV) и графики (PNG) всех экспериментов, run_log.txt"],
        ["report_pdf/", "Готовый PDF-отчёт"],
    ], col_widths=[60 * mm, 110 * mm]))

    OUT_DIR.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title="Лабораторная работа №4", author="Ильин А.А.")
    doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
    print("Saved:", OUT)


if __name__ == "__main__":
    build()
