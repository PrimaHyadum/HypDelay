# HypDelay – библиотека для численного решения гиперболических ДУ с запаздыванием
**HypDelay** – это Python-библиотека для численного решения гиперболических дифференциальных уравнений в частных производных с постоянным запаздыванием. Она предоставляет единообразный интерфейс для задания моделей, численных методов и вычисления решений, а также включает инструменты для анализа ошибок, визуализации и анимации, в том числе через графический интерфейс.
- **Версия:** 0.1.0
- **Зависимости:** `numpy`, `matplotlib`, `scipy`, `tkinter`.
## Оглавление
1. [Установка](#установка)
2. [Архитектура и основные компоненты](#архитектура-и-основные-компоненты)
3. [Быстрый старт: первое решение](#быстрый-старт-первое-решение)
4. [Модели](#модели)
- [HyperbolicModel](#hyperbolicmodel)
- [SineGordonKinkModel](#sinegordonkinkmodel)
- [Phi4KinkModel](#phi4kinkmodel)
- [DoubleSineGordonKinkModel](#doublesinegordonkinkmodel)
5. [Численные методы](#численные-методы)
6. [Солвер `DelayPDESolver`](#солвер-delaypdesolver)
7. [Конфигурация `Config`](#конфигурация-config)
8. [Вычисление ошибок](#вычисление-ошибок)
9. [Визуализация решения](#визуализация-решения)
10. [Анимация решений](#анимация-решений)
11. [Сохранение результатов](#сохранение-результатов)
12. [Графический интерфейс (GUI)](#графический-интерфейс-gui)
13. [Создание пользовательских моделей и методов](#создание-пользовательских-моделей-и-методов)
14. [Примеры использования](#примеры-использования)
## Установка
```bash
pip install hypdelay # если пакет загружен в PyPI
````
Или установите из исходников:

```bash
git clone https://github.com/yourusername/hypdelay.git
cd hypdelay
pip install -e .
```
## Архитектура и основные компоненты

```text
hypdelay/
├── __init__.py
├── core/
│ ├── solver.py – класс DelayPDESolver
│ ├── history.py – работа с историей
│ ├── config.py – класс Config
│ └── base.py – абстрактные классы PDEModel, PDEMethod
├── models/ – встроенные модели (Hyperbolic, Sine‑Gordon, φ⁴ и др.)
├── methods/ – методы интерполяции/экстраполяции
├── utils/ – вычисление ошибок, построение графиков, анимация, сохранение, устойчивость
└── gui/ – графический интерфейс на tkinter
```
Ключевые сущности:

* **`PDEModel`** – абстрактный класс, описывающий уравнение (точное решение, начальные/граничные условия, нелинейный источник).
* **`PDEMethod`** – абстрактный класс, реализующий способ вычисления экстраполированного значения `u_current` и запаздывающего `u_delayed`.
* **`DelayPDESolver`** – основной солвер, объединяет модель, метод и параметры сетки.
* **`Config`** – хранит все параметры задачи и вычисляемые величины.
* **Утилиты** – функции для вычисления ошибок, построения графиков, анимации, сохранения.

## Быстрый старт: первое решение

Пример решения гиперболического тестового уравнения (модель `HyperbolicModel`):

```python
import hypdelay as hd
# Модель
model = hd.HyperbolicModel(a=1.0, tau=2.0)
# Метод
method = hd.PrimeMethod()
# Солвер
solver = hd.DelayPDESolver(
model=model, method=method,
x0=0.0, t0=0.0, s=0.4, L=1.0, T=4.0, M=18, N=8
)
# Решение
A, cfg = solver.solve()
# Ошибки
max_err, l2_err, mae = hd.compute_error(A, cfg)
# Визуализация
hd.plot_solution(model, A, cfg)
```
## Модели

Все модели наследуются от `PDEModel` и обязаны реализовать методы `exact_solution`, `initial_derivative`, `source_term`. Дополнительно можно переопределить `initial_condition`, `boundary_condition_x0`, `boundary_condition_L`.

Ниже приведены встроенные модели с их математическими описаниями.

### HyperbolicModel

**Уравнение:**

$$
\frac{\partial^2 u}{\partial t^2} = a^2 \frac{\partial^2 u}{\partial x^2} + a^2 \pi^2 e^{-t} \sin(\pi x) + e^{\tau - 2t} \sin^2(\pi x) + u(x,t)\bigl(1 - u(x, t-\tau)\bigr)
$$

**Начальные условия:**

$$
u(x, t) = e^{-t}\sin(\pi x), \quad -\tau \le t \le 0,\; 0 \le x \le 1.
$$

**Граничные условия:**

$$
u(0, t) = 0, \quad u(1, t) = 0, \quad 0 \le t \le T.
$$

**Точное решение:**

$$
u(x,t) = e^{-t}\sin(\pi x).
$$

**Пример вызова:**

```python
model = hd.HyperbolicModel(a=1.0, tau=2.0)
```
Параметры:

* `a` – коэффициент при второй пространственной производной.
* `tau` – постоянное запаздывание.

---

### SineGordonKinkModel

**Уравнение:**

$$
\frac{\partial^2 u}{\partial t^2} = a^2 \frac{\partial^2 u}{\partial x^2} - \sin\bigl(u(x, t-\tau)\bigr).
$$

**Аналитическое решение (кинк):**

$$
u(x,t) = 4\arctan!\Bigl[\exp\bigl(m\,\gamma\,(x - v t) + \delta\bigr)\Bigr], \qquad \gamma = \frac{1}{\sqrt{1-v^2}}.
$$

Параметры кинка: `m` (форма), `u_kin=v` (скорость, $|v|<1$), `delta_kin=δ` (сдвиг).

**Начальные и граничные условия** согласованы с точным решением.

**Пример вызова:**

```python
model = hd.SineGordonKinkModel(a=1.0, tau=0.0, m=1.0, u_kin=0.5, delta_kin=0.0)
```
---

### Phi4KinkModel

**Уравнение φ⁴:**

$$
\frac{\partial^2 u}{\partial t^2} = a^2 \frac{\partial^2 u}{\partial x^2} - m^2\bigl(u^3(x,t-\tau) - u(x,t)\bigr).
$$

**Аналитическое решение (кинк):**

$$
u(x,t) = \tanh!\Bigl(\frac{m}{\sqrt{2}}\,\gamma\,(x - v t) + \delta\Bigr), \qquad \gamma = \frac{1}{\sqrt{1-v^2}}.
$$

**Пример вызова:**

```python
model = hd.Phi4KinkModel(a=1.0, tau=0.0, m=1.0, u_kin=0.5, delta_kin=0.0)
```
---

### DoubleSineGordonKinkModel

**Уравнение двойного синус-Гордона:**

$$
\frac{\partial^2 u}{\partial t^2} = a^2 \frac{\partial^2 u}{\partial x^2} - \sin\bigl(u(x,t)\bigr) - \lambda \sin!\bigl(2u(x,t-\tau)\bigr).
$$

**Аналитическое решение (кинк):**

$$
u(x,t) = 2\arctan!\Bigl[ A \tanh!\bigl(c\,(x - v t) + \delta\bigr) \Bigr],
$$
$$
A = \sqrt{\frac{1}{1+2\lambda}}, \qquad c = \frac{\sqrt{1+2\lambda}}{2}\,m\,\gamma, \qquad \gamma = \frac{1}{\sqrt{1-v^2}}.
$$

**Пример вызова:**

```python
model = hd.DoubleSineGordonKinkModel(a=1.0, tau=0.0, m=1.0, u_kin=0.5, delta_kin=0.0, lam=0.5)
```
## Численные методы

Все методы наследуются от `PDEMethod` и реализуют два обязательных метода:

* `get_current_extrapolation(cfg, u_prev, u_prev2)` – возвращает экстраполированное значение для текущего временного слоя.
* `get_delayed_value(cfg, history, A, t_current, x_idx)` – возвращает значение с запаздыванием $u(x, t-\tau)$.

Встроенные методы:

| Класс | Описание |
| --- | --- |
| `PWLinearInterpExtrap` | Кусочно-линейная интерполяция и экстраполяция продолжением (формулы (7)–(8) из теории). |
| `PrimeMethod` | Прямой доступ по индексу без учёта предыстории. Воспроизводит исходный метод из `start_late_4.py`. |
| `PrimeHybridMethod` | Прямой доступ по индексу с учётом предыстории. |

**Пример создания:**

```python
method = hd.PrimeMethod()
```
## Солвер `DelayPDESolver`

Класс `DelayPDESolver` объединяет модель, метод и параметры сетки. Конструктор:

```python
solver = hd.DelayPDESolver(
model, method,
x0=0.0, t0=0.0, s=0.4, L=1.0, T=4.0, M=18, N=8
)
```
Параметры:

* `x0`, `t0` – начальные координаты.
* `s` – весовой коэффициент схемы (0 – явная, 1 – неявная).
* `L`, `T` – конечные координаты.
* `N` – число узлов по пространству.
* `M` – число узлов по времени.

Метод `solve(verbose=False)` возвращает кортеж `(A, cfg)`, где `A` – матрица решения размера `(M, N)`, `cfg` – объект конфигурации `Config`.

## Конфигурация `Config`

Объект `Config` содержит все параметры задачи и расчётные величины:

* От модели: `model`, `a`, `tau`
* От метода: `method`
* Параметры сетки: `x0`, `t0`, `L`, `T`, `N`, `M`, `s`
* Вычисляемые: `h` (шаг по x), `delta` (шаг по t), `rh` (`(a*delta/h)^2`), `r` (`s * rh`), `eps=1e-12`.

Доступ к полям: `cfg.h`, `cfg.delta` и т.д.

## Вычисление ошибок

Функция `compute_error(A, cfg)` сравнивает численное решение с точным (из модели) и возвращает три метрики:

```python
max_err, l2_err, mae = hd.compute_error(A, cfg)
```
* `max_err` – максимальная абсолютная ошибка.
* `l2_err` – $L^2$-норма ошибки.
* `mae` – средняя абсолютная ошибка.

## Визуализация решения

```python
hd.plot_solution(model, A, cfg, title="Численное решение", show=True)
```
Строит 3D-поверхность численного решения и, если доступно, точного решения. Параметр `show=False` позволяет отложить показ (например, для встраивания в GUI).

## Анимация решений

Класс `PlotAnimation` создаёт анимацию смены 3D-поверхностей для набора решений.

```python
solutions = [(A1, cfg1), (A2, cfg2), ...] # список кортежей
anim = hd.PlotAnimation(
solutions,
fps=5, # кадры в секунду
figsize=(10,7), # размер фигуры в дюймах
dpi=100, # разрешение
plot_type='surface', # 'surface' или 'wireframe'
elev=30, azim=-60, # начальный угол обзора
repeat=True, # зацикливание
errors=None # список кортежей (max_err, l2_err, mae) для отображения
)
anim.create_animation() # подготовка анимации
anim.show() # показ в окне matplotlib
```
Сохранение в файл:

```python
anim.save("path/to/animation.gif") # GIF (требуется ImageMagick)
anim.save("path/to/animation.mp4", writer='ffmpeg') # MP4 (требуется FFmpeg)
```
**Примечание:** В GUI анимация запускается через окно `AnimationDialog`, где можно настроить FPS, тип графика и ракурс, а также сохранить файл.

## Сохранение результатов

Функция `save_all_results` записывает CSV с параметрами и, опционально, ошибками:

```python
# Только параметры
hd.save_all_results([cfg1, cfg2, ...], "params.csv")
# С ошибками
results = [(cfg1, {'max_err': 0.1, 'l2_err': 0.05, 'mae': 0.01}), ...]
hd.save_all_results(results, "results.csv")
```
## Графический интерфейс (GUI)

Для запуска GUI:

```python
import hypdelay as hd
app = hd.run_gui()
```
Основные возможности:

* Выбор модели и численного метода.
* Ввод параметров модели и схемы (с подсказками).
* Расчёт одного решения с визуализацией и метриками.
* Параметрический перебор с фильтрацией по устойчивости и ошибкам.
* Анимация полученных решений (кнопка «Анимация» в окне перебора).
* Загрузка параметров из CSV.

Интерфейс реализован на `tkinter` и `matplotlib`.

## Создание пользовательских моделей и методов

### Новая модель

Унаследуйте от `PDEModel` и реализуйте обязательные методы:

```python
from hypdelay import PDEModel
import numpy as np
class MyModel(PDEModel):
def __init__(self, a, tau, extra_param):
super().__init__(a, tau)
self.extra = extra_param
def exact_solution(self, x, t):
return np.sin(np.pi * x) * np.cos(t)
def initial_derivative(self, x, t):
return -np.sin(np.pi * x) * np.sin(t)
def source_term(self, u_current, u_delayed, x, t):
# f(x,t,u,u_tau)
return -np.pi**2 * self.a**2 * u_current + u_delayed * (1 - u_current)
# Необязательно:
def initial_condition(self, x, t):
return self.exact_solution(x, t)
def boundary_condition_x0(self, x, t):
return 0.0
def boundary_condition_L(self, x, t):
return 0.0
```
### Новый метод

Унаследуйте от `PDEMethod` и реализуйте:

```python
from hypdelay import PDEMethod
class MyMethod(PDEMethod):
def get_current_extrapolation(self, cfg, u_prev, u_prev2):
delta = cfg.delta
s = cfg.s
return ((delta + s) * u_prev - s * u_prev2) / delta
def get_delayed_value(self, cfg, history, A, t_current, x_idx):
tau = cfg.tau
t0 = cfg.t0
delta = cfg.delta
t_delayed = t_current - tau
if t_delayed < t0:
return None # u_delayed = u_current
idx = int(round((t_delayed - t0) / delta))
idx = max(0, min(idx, len(A)-1))
return A[idx][x_idx]
```
## Примеры использования

В репозитории есть файлы `test1.py` и `test2.py`, демонстрирующие:

* Решение с встроенными моделью и методом.
* Создание пользовательских модели и метода, аналогичных описанным выше.
* Параметрический перебор по `tau` и создание анимации.

Запустите их:

```bash
python test1.py
python test2.py
```
или откройте GUI:

```bash
python -c "import hypdelay as hd; hd.run_gui()"
```
---

**Авторы:** [Ваше имя]
**Благодарности:** Научному руководителю за постановку задачи и ценные обсуждения.

```text
Этот документ покрывает все указанные вами аспекты: описание библиотеки, все функции, модели с математическими формулами, примеры вызова, архитектуру, GUI, сохранение, анимацию и инструкцию для пользователя. Вы можете добавить его в корень репозитория как `README.md` или разместить в отдельном файле документации.
```
