# -*- coding: utf-8 -*-
"""Поясняющие схемы для конспекта (нейрон, активации, градиентный спуск и т.д.)."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


# 1. Схема нейрона ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
inputs = [("x₁ (вес = 84)", 3.2), ("x₂ (рост = 180)", 2.0), ("1 (смещение)", 0.8)]
weights = ["w₁ = 0.3", "w₂ = 0.1", "b = −39"]
for (txt, yy), w in zip(inputs, weights):
    ax.add_patch(Circle((1.2, yy), 0.32, color="#cfe2f3", ec="#23445f"))
    ax.text(0.2, yy + 0.45, txt, fontsize=10)
    ax.add_patch(FancyArrowPatch((1.55, yy), (4.3, 2.0), arrowstyle="->",
                                 mutation_scale=14, color="#23445f"))
    ax.text(2.4, (yy + 2.0) / 2 + 0.15, w, fontsize=10, color="#b45f06")
ax.add_patch(Circle((4.8, 2.0), 0.55, color="#fce5cd", ec="#b45f06"))
ax.text(4.8, 2.0, "Σ", fontsize=20, ha="center", va="center")
ax.text(4.8, 1.0, "r = w₁x₁ + w₂x₂ + b", ha="center", fontsize=10)
ax.add_patch(FancyArrowPatch((5.4, 2.0), (6.5, 2.0), arrowstyle="->",
                             mutation_scale=14))
ax.add_patch(Rectangle((6.5, 1.5), 1.6, 1.0, color="#d9ead3", ec="#38761d"))
ax.text(7.3, 2.0, "σ(r)", fontsize=16, ha="center", va="center")
ax.text(7.3, 1.1, "активация", ha="center", fontsize=10)
ax.add_patch(FancyArrowPatch((8.1, 2.0), (9.2, 2.0), arrowstyle="->",
                             mutation_scale=14))
ax.text(9.3, 2.0, "y", fontsize=16, va="center")
ax.text(8.7, 2.5, "0.985 → «male»", fontsize=10, ha="center")
ax.set_title("Искусственный нейрон: взвешенная сумма + функция активации")
save(fig, "neuron.png")

# 2. Функции активации ----------------------------------------------------
x = np.linspace(-6, 6, 400)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
funcs = [
    ("sigmoid: 1/(1+e⁻ˣ)", 1 / (1 + np.exp(-x))),
    ("tanh(x)", np.tanh(x)),
    ("ReLU: max(0, x)", np.maximum(0, x)),
]
for ax, (t, yv) in zip(axes[:3], funcs):
    ax.plot(x, yv, lw=2.5); ax.set_title(t); ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
s = 1 / (1 + np.exp(-x))
axes[0].plot(x, s * (1 - s), "--", label="производная (max 0.25)")
axes[0].legend(fontsize=8, loc="upper left")
logits = np.array([2.0, 1.0, 0.1, -1.0])
p = np.exp(logits) / np.exp(logits).sum()
axes[3].bar(["z₁=2", "z₂=1", "z₃=0.1", "z₄=−1"], p, color="#6fa8dc")
for i, v in enumerate(p):
    axes[3].text(i, v + 0.01, f"{v:.2f}", ha="center")
axes[3].set_title("softmax: eᶻⁱ / Σeᶻʲ (сумма = 1)")
save(fig, "activations.png")

# 3. Архитектура сети MNIST ----------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
layers = [("Вход\n784 пикселя", 1, 9, "#cfe2f3"), ("Скрытый 1\n64, sigmoid", 3.5, 6, "#fce5cd"),
          ("Скрытый 2\n64, sigmoid", 6, 6, "#fce5cd"), ("Выход\n10, softmax", 8.5, 5, "#d9ead3")]
pos = []
for name, xx, n, col in layers:
    ys = np.linspace(0.6, 4.2, n)
    pos.append((xx, ys))
    for yy in ys:
        ax.add_patch(Circle((xx, yy), 0.16, color=col, ec="0.3"))
    ax.text(xx, 4.65, name, ha="center", fontsize=10)
for (x1, y1), (x2, y2) in zip(pos[:-1], pos[1:]):
    for a in y1:
        for b in y2:
            ax.plot([x1 + 0.16, x2 - 0.16], [a, b], color="0.75", lw=0.4, zorder=0)
ax.text(2.25, 0.1, "784·64+64 = 50 240", ha="center", fontsize=9)
ax.text(4.75, 0.1, "64·64+64 = 4 160", ha="center", fontsize=9)
ax.text(7.25, 0.1, "64·10+10 = 650", ha="center", fontsize=9)
ax.set_title("Полносвязная сеть из методички: всего 55 050 обучаемых параметров")
save(fig, "mlp.png")

# 4. Градиентный спуск ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
w = np.linspace(-3, 3, 300)
L = w ** 2
axes[0].plot(w, L, lw=2)
pts = [2.6]
for _ in range(6):
    pts.append(pts[-1] - 0.3 * 2 * pts[-1])
pts = np.array(pts)
axes[0].plot(pts, pts ** 2, "o-", color="tab:red")
axes[0].set_title("Градиентный спуск: w ← w − η·∂L/∂w")
axes[0].set_xlabel("вес w"); axes[0].set_ylabel("ошибка L(w)")
for lr, c, lab in [(0.05, "tab:blue", "η мала: медленно"), (0.3, "tab:green", "η норм."),
                   (1.02, "tab:red", "η велика: расходится")]:
    v = [2.5]
    for _ in range(10):
        v.append(v[-1] - lr * 2 * v[-1])
    axes[1].plot(np.abs(v), "o-", ms=3, color=c, label=lab)
axes[1].set_yscale("log"); axes[1].set_xlabel("шаг"); axes[1].set_ylabel("|w − w*|")
axes[1].set_title("Влияние скорости обучения η"); axes[1].legend()
save(fig, "gradient_descent.png")

# 5. Оптимизаторы на «овраге» --------------------------------------------
def f_grad(p):
    x, y = p
    return np.array([0.2 * x, 4 * y])

def run(kind, steps=60):
    p = np.array([-4.5, 1.5]); path = [p.copy()]
    v = np.zeros(2); s = np.zeros(2); m = np.zeros(2); g2 = np.zeros(2)
    for t in range(1, steps + 1):
        g = f_grad(p)
        if kind == "SGD":
            p = p - 0.2 * g
        elif kind == "Momentum":
            v = 0.8 * v - 0.12 * g; p = p + v
        elif kind == "RMSprop":
            s = 0.9 * s + 0.1 * g ** 2; p = p - 0.15 * g / (np.sqrt(s) + 1e-8)
        elif kind == "Adam":
            m = 0.9 * m + 0.1 * g; g2 = 0.999 * g2 + 0.001 * g ** 2
            mh = m / (1 - 0.9 ** t); vh = g2 / (1 - 0.999 ** t)
            p = p - 0.12 * mh / (np.sqrt(vh) + 1e-8)
        path.append(p.copy())
    return np.array(path)

X1, Y1 = np.meshgrid(np.linspace(-5, 1.5, 200), np.linspace(-2, 2, 200))
Z = 0.1 * X1 ** 2 + 2 * Y1 ** 2
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.contour(X1, Y1, Z, levels=20, cmap="Greys", linewidths=0.6)
for kind, c in [("SGD", "tab:red"), ("Momentum", "tab:orange"),
                ("RMSprop", "tab:green"), ("Adam", "tab:blue")]:
    pth = run(kind)
    ax.plot(pth[:, 0], pth[:, 1], "o-", ms=2.5, lw=1.2, color=c, label=kind)
ax.plot(0, 0, "k*", ms=14)
ax.set_title("Путь разных оптимизаторов к минимуму (★) в вытянутом «овраге»")
ax.legend()
save(fig, "optimizers_path.png")

# 6. Переобучение / недообучение -----------------------------------------
rng = np.random.default_rng(1)
xs = np.sort(rng.uniform(0, 1, 15)); ys = np.sin(2 * np.pi * xs) + rng.normal(0, 0.2, 15)
xx = np.linspace(0, 1, 300)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, deg, t in zip(axes, (1, 4, 14), ("Недообучение (степень 1)",
                                          "Хорошо (степень 4)",
                                          "Переобучение (степень 14)")):
    coef = np.polyfit(xs, ys, deg)
    ax.scatter(xs, ys, color="k", s=15)
    ax.plot(xx, np.clip(np.polyval(coef, xx), -2, 2), lw=2)
    ax.plot(xx, np.sin(2 * np.pi * xx), "--", color="0.6")
    ax.set_title(t); ax.set_ylim(-2, 2)
save(fig, "overfitting.png")

# 7. KFold vs TimeSeriesSplit --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
n = 20
for k in range(5):
    test = np.zeros(n); test[k * 4:(k + 1) * 4] = 1
    for i in range(n):
        axes[0].add_patch(Rectangle((i, k), 0.95, 0.8,
                                    color="tab:orange" if test[i] else "tab:blue"))
    end = 4 + k * 3
    for i in range(n):
        col = "tab:blue" if i < end else ("tab:orange" if i < end + 3 else "0.9")
        axes[1].add_patch(Rectangle((i, k), 0.95, 0.8, color=col))
for ax, t in zip(axes, ("KFold: тест может быть «в прошлом» → утечка",
                        "TimeSeriesSplit: учим на прошлом, проверяем на будущем")):
    ax.set_xlim(0, n); ax.set_ylim(0, 5); ax.set_title(t, fontsize=10)
    ax.set_xlabel("время →"); ax.set_yticks([]); ax.grid(False)
axes[0].set_ylabel("фолд")
save(fig, "cv_schemes.png")

# 8. Затухающий градиент --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.3))
depth = np.arange(1, 8)
ax.bar(depth, 0.25 ** depth, color="#e06666")
ax.set_yscale("log")
ax.set_xlabel("число сигмоидных слоёв между ошибкой и весом")
ax.set_ylabel("макс. множитель градиента")
ax.set_title("Затухание градиента: σ'(x) ≤ 0.25, множители перемножаются")
for d in depth:
    ax.text(d, 0.25 ** d * 1.3, f"{0.25 ** d:.0e}", ha="center", fontsize=8)
save(fig, "vanishing_gradient.png")

# 9. Компоненты временного ряда ------------------------------------------
t = np.arange(0, 3 * 365)
trend = 1300 + 0.03 * t
season = 120 * np.cos(2 * np.pi * t / 365)
week = -60 * (np.arange(len(t)) % 7 >= 5)
noise = np.random.default_rng(0).normal(0, 25, len(t))
fig, axes = plt.subplots(5, 1, figsize=(11, 7), sharex=True)
for ax, (yv, name) in zip(axes, [(trend + season + week + noise, "ряд = сумма компонент"),
                                  (trend, "тренд"), (season, "годовая сезонность"),
                                  (week, "недельная сезонность (выходные ниже)"),
                                  (noise, "шум")]):
    ax.plot(t, yv, lw=0.8); ax.set_ylabel(name, rotation=0, ha="right", fontsize=9)
axes[-1].set_xlabel("день")
save(fig, "ts_components.png")

print("Figures saved to", OUT)
