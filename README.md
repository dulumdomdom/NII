# Лабораторные работы №3 и №4 по искусственному интеллекту

| Работа | Тема | Решение | Отчёт |
|---|---|---|---|
| ЛР3 | Регрессионный анализ временных рядов: прогноз энергопотребления Германии (OPSD) | [lab3/lab3_solution.py](lab3/lab3_solution.py) | [PDF](lab3/report_pdf/Лабораторная_работа_3.pdf) |
| ЛР4 | Введение в нейронные сети: классификация цифр MNIST (Keras) | [lab4/lab4_solution.py](lab4/lab4_solution.py) | [PDF](lab4/report_pdf/Лабораторная_работа_4.pdf) |

**[Конспект_ЛР3-ЛР4.pdf](Конспект_ЛР3-ЛР4.pdf)** — общий конспект к защите обеих работ: основы МО, теория,
разбор кода методичек, результаты и вопросы с ответами.

## Основные результаты

**ЛР3.** Лучшая модель — KNN: средний R² = 0.864 на горизонтах 1–12 месяцев; прогноз на 2017 год —
R² = 0.834, MAE = 55.2 ГВт·ч (~4% от среднего потребления).

**ЛР4** (сеть 784–64–64–10, sigmoid, 10 эпох, среднее по 3 запускам):

- число скрытых слоёв 1→5: с Adam точность 0.957–0.960 при 1–4 слоях и 0.945 при 5; с SGD падает с 0.920 до 0.136 (затухание градиента);
- доля теста 5%→50%: точность снижается с 0.968 до 0.956 (меньше обучающих данных);
- оптимизаторы: RMSprop / SGD+momentum / Nadam / Adam / Adamax ≈ 0.96, SGD 0.909, Adagrad 0.755, Adadelta 0.410.

## Запуск

```bash
# ЛР3 (Python 3.10)
cd lab3
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python lab3_solution.py
.venv/Scripts/python make_pdf_report.py

# ЛР4 (Python 3.10, TensorFlow 2.19)
cd lab4
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python lab4_solution.py          # полный прогон ~1 час на CPU
.venv/Scripts/python lab4_solution.py --quick  # быстрый прогон
.venv/Scripts/python make_pdf_report.py

# Конспект (использует окружение ЛР4)
cd konspekt
../lab4/.venv/Scripts/python make_figures.py
../lab4/.venv/Scripts/python make_konspekt.py
```

## Структура

```
lab3/        решение ЛР3, методичка (.ipynb), данные, результаты, отчёт и шпаргалка
lab4/        решение ЛР4, результаты (CSV/PNG, run_log.txt), отчёт
konspekt/    генератор общего конспекта и поясняющие схемы
Л_р_№4_Введение_в_нейронные_сети.ipynb   методичка ЛР4
```
