"""
В этом файле более подробно расскажу про шаблоны PDEModel и PDEMethod
Каждый из этих шаблонов строго ориентирован, то есть
некоторые функции необходимо прописывать
"""

import hypdelay as hd
from hypdelay import PDEMethod, PDEModel
from hypdelay import HyperbolicModel, SineGordonKinkModel
import numpy as np

"""
Подробнее о PDEModel
PDEModel - это абстрактный класс, то есть при наследовании
необходимо будет изменить некоторые функции, которые я опишу далее
"""

"""
class ModelTest(PDEModel):
ModelTest - название класса
ModelTest(PDEModel) - наследование от шаблона PDEModel
"""
class ModelTest(PDEModel):
    """
    в __init__(self, {где-то здесь})
    за место {} - мы пишем обязательно a, tau,
    а далее произвольные параметры, которые
    мы будем в дальнейшем использовать в модели
    """
    def __init__(self, a, tau, param1, param2):
        # Эта строчка просто должна быть именно в таким формате
        super().__init__(a, tau)

        # self. - обозначение для класса, что этот параметр можно использовать во всем классе где угодно
        # но вызвать этот параметр можно только через self.param1
        self.param1 = param1
        self.param2 = param2

    def exact_solution(self, x, t):
        """
        ОБЯЗАТЕЛЬНАЯ ФУНКЦИЯ

        Здесь будет точное решение
        Так записывается u(x, t) = exp(-t) * sin(pi * x)

        Параметры x, t передаются от других классов при вызове
        """
        u = np.exp(-t) * np.sin(np.pi * x)
        return u

    def initial_derivative(self, x, t):
        """
        ОБЯЗАТЕЛЬНАЯ ФУНКЦИЯ

        Производная du/dt(x, t) для построения второго временного слоя.
        Производную вычисляем с переменной t, так как t0 может быть не равной нулю

        для du/dt(x, t) = -exp(-t) * sin(pi * x)
        """
        du_dt = -np.exp(-t) * np.sin(np.pi * x)
        return -np.exp(-t) * np.sin(np.pi * x)

    def source_term(self, u_current, u_delayed, x, t):
        """
        ОБЯЗАТЕЛЬНАЯ ФУНКЦИЯ

        Нелинейный источник f(x, t, u(x, t), u_t(x, *))

        u_current - Значение u(x, t), вычисляемое через PDEMethod.get_current_extrapolation
        u_delayed - Постоянное запаздывание u(x, t - tau), вычисляемое через PDEMethod.get_delayed_value
        x - Переменная по пространству
        t - Переменная по времени

        для уравнения (13) из презентации это будет ниже
        """
        func = self.a ** 2 * np.pi ** 2 * np.exp(-t) * np.sin(np.pi * x)
        func += np.exp(self.tau - 2 * t) * np.sin(np.pi * x) ** 2
        func += u_current * (1.0 - u_delayed)
        return func

    """ДАЛЕЕ ИДУТ НЕ ОБЯЗАТЕЛЬНЫЕ ФУНКЦИИ"""

    def boundary_condition_x0(self, x, t):
        """
        Возвращает граничное условие для x=x0, t in [t0, T]
        Изначально возвращает exact_solution(x, t)
        Если его не прописать в новом классе

        Но если переписать, то можно изменить граничное условие, к примеру ниже
        Тут мы возвращаем self.param1 при любом значении x и t
        """
        return self.param1

    def boundary_condition_L(self, x, t):
        """
        Возвращает граничное условие для x=L, t in [t0, T]
        Изначально возвращает exact_solution(x, t)

        Аналогично boundary_condition_x0, только здесь x=L
        Можем его также переписать
        """
        return self.param2

    def initial_condition(self, x, t):
        """
        Начальное условие: u(x,t) при t ∈ [-τ, 0].
        Изначально возвращает exact_solution(x, t).

        Мы также можем переопределить данную функцию
        Возьмем начальные условия из 11 слайда презентации
        """
        u = np.exp(-t) * np.sin(np.pi * x)
        return u

    """
    в классе еще также присутствуют две другие функции:
    description и param_info, но они необходимы, чтобы определить их в интерфейсе
    поэтому их смело можно пропускать
    """

model = ModelTest(a=2.0, tau=4.0, param1=0.0, param2=0.0)

"""
PDEMethod - это метод решения для u(x, t) и u(x, t - tau)
При вычислении B[j - 1] слоя

содержит в себе всего 2 функции: 
get_current_extrapolation - подразумевает под собой экстраполяцию для u_current в модели
get_delayed_value - подразумевает собой интерполяцию для u_delayed

и обе эти функции - обязательные для переопределения
"""

"""
Также наследуем шаблон, но теперь от PDEMethod

MethodTest - полная копия вашего решения из start_late_4.py
"""
class MethodTest(PDEMethod):
    def get_current_extrapolation(self, cfg, u_prev, u_prev2):
        """
        В данной функции мы вычисляем значение u(x, t)
        Если рассматривать экстраполяцию продолжением, формула (8)
        то ее запись будет выглядеть так

        если вычисляемое значение находится на слое k + 1, то
        u_prev - значение U на прошлом слое, слой k
        u_prev2 - значение U на позапрошлом слой k - 1
        """
        # через cfg можем получить любую константу
        delta = cfg.delta
        s = cfg.s
        extrapolation = ((delta + s) * u_prev - s * u_prev2) / delta
        return extrapolation

    def get_delayed_value(self, cfg, history, A, t_current, x_idx):
        """
        В данной функции мы вычисляем значение u(x, t - tau)
        В идеале - здесь проводится интерполяция,
        но можно использовать разные подходы
        для вычисления u(x, t - tau)

        Вот описание каждого параметра:
        cfg - конфиг, содержащий в себе параметры
        history - экземляр класса History, который вычисляет значение для поредыстории
        A - матрица уже существующих решений
        t_current - время, которое сейчас вычисляется, u(x, t_current - tau)
        x_idx - индекс пространственной переменной, на которой сейчас вычисляется запаздывание
        """

        # забираем из cfg необходимые параметры
        tau = cfg.tau
        t0 = cfg.t0
        delta = cfg.delta

        # t_delayed - вычисленное вермя с запаздыванием
        t_delayed = t_current - tau

        # Если время с запаздыванием меньше нуля
        if t_delayed < t0:

            return None  # специальное значение, означающее, что u_delayed = u_current

        # Если время с запаздыванием больше, либо равно нулю
        else:
            idx = int(round((t_delayed - t0) / delta))
            idx = max(0, min(idx, len(A) - 1))
            return A[idx][x_idx]

"""
Переопределив класс метода
можно сделать его экземпляр
"""
method = MethodTest()

"""
Далее - все точно то же самое, что и в test1.py
Но теперь параметры взяты с 11 слайда презентации
"""

x0 = 0.0
t0 = 0.0
s = 0.4
L = 1.0
T = 6.0
M = 16
N = 29

solver = hd.DelayPDESolver(model=model, method=method, x0=x0, t0=t0,
                           s=s, L=L, T=T, M=M, N=N)
A, cfg = solver.solve()
max_error, l2_error, MAE = hd.compute_error(A, cfg)
hd.plot_solution(model=model, A=A, cfg=cfg)

print("ПАРАМЕТРЫ:")
print(f'a = {cfg.a}, tau = {cfg.tau}, h = {cfg.h}, delta = {cfg.delta}')
print(f'T = {cfg.T}, L = {cfg.L}, N = {cfg.N}, M = {cfg.M}')
print()
print("МЕТРИКИ:")
print(f"Максимальная ошибка: {max_error:8f}")
print(f"L2 ошибка: {l2_error:8f}")
print(f"MAE: {MAE:8f}")
