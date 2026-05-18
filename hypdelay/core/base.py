"""абстрактные классы PDEModel, PDEMethod"""
from abc import ABC, abstractmethod
import numpy as np

class PDEMethod(ABC):

    @abstractmethod
    def get_current_extrapolation(self, cfg, u_prev, u_prev2):
        """
        Возвращает экстраполированное значение u_current (на новом слое)
        по двум предыдущим слоям (u_prev – слой k, u_prev2 – слой k-1).
        """
        pass

    @abstractmethod
    def get_delayed_value(self, cfg, history, A, t_current, x_idx):
        """
        Возвращает u(x_idx, t_current - tau).
        - history: объект History (или массив) для t < 0.
        - computed_layers: список уже вычисленных слоёв для t >= 0 (индексы 0,1,...,k).
        - t_current: текущее время (для которого ищем решение, может быть >0).
        - tau: запаздывание.
        - x_idx: индекс по пространству.
        - delta: шаг по времени.
        """
        pass


class PDEModel(ABC):
    """
    Абстрактный класс уравнения гиперболического типа с запаздыванием:

        u_tt = a^2 * u_xx + f(x, t, u(x, t), u_t(x, t - tau))

    :params:
    :param a: Константа гиперболического уравнения
    :param tau: Запаздывание

    """
    def __init__(self, a: float = 1.0, tau: float = 0.0):
        self.a = a
        self.tau = tau

    @abstractmethod
    def exact_solution(self, x, t):
        """
        Точное (аналитическое) решение u(x,t).
        Используется для задания начальных/граничных условий и оценки погрешности.

        :params:
        :param x: Переменная по пространству
        :param t: Переменная по времени
        """
        pass

    @abstractmethod
    def initial_derivative(self, x, t):
        """
        Производная du/dt(x, t) для построения второго временного слоя.

        :params:
        :param x: Переменная по пространству
        :param t: Переменная по времени
        """
        pass

    @abstractmethod
    def source_term(self, u_current, u_delayed, x, t):
        """
        Нелинейный источник f(x, t, u(x, t), u_t(x, *))

        :params:
        :param u_current: Значение u(x, t), вычисляемое через PDEMethod.get_current_extrapolation
        :param u_delayed: Постоянное запаздывание u(x, t - tau), вычисляемое через PDEMethod.get_delayed_value
        :param x: Переменная по пространству
        :param t: Переменная по времени
        """
        pass

    def boundary_condition_x0(self, x, t):
        """
        Возвращает граничное условие для x=x0, t in [t0, T]
        Изначально возвращает exact_solution(x, t)

        :params:
        :param x: Переменная по пространству
        :param t: Переменная по времени
        """
        return self.exact_solution(x, t)

    def boundary_condition_L(self, x, t):
        """
        Возвращает граничное условие для x=L, t in [t0, T]
        Изначально возвращает exact_solution(x, t)

        :params:
        :param x: Переменная по пространству
        :param t: Переменная по времени
        """
        return self.exact_solution(x, t)

    def initial_condition(self, x, t):
        """
        Начальное условие: u(x,t) при t ∈ [-τ, 0].
        Обычно совпадает с exact_solution(x, t).

        :params:
        :param x: Переменная по пространству
        :param t: Переменная по времени
        """
        return self.exact_solution(x, t)

    @classmethod
    def description(cls):
        """Возвращает строку с уравнением и аналитическим решением для GUI."""
        return "Уравнение не задано."

    @classmethod
    def param_info(cls):
        """Возвращает список (имя, значение_по_умолчанию, описание) для GUI и CSV."""
        return []  # переопределить в наследниках