from hypdelay.core.base import PDEModel
import numpy as np

class SineGordonKinkModel(PDEModel):
    """
    Уравнение синус-Гордона: u_tt - a² u_xx + sin(u) = 0.

    Аналитическое решение – кинк:

        u(x,t) = 4·arctan( exp( m·γ·(x - v·t) + δ ) ),   γ = 1/√(1 - v²)

    Параметры кинка:
        m         – параметр формы (обычно 1)
        u_kin (v) – скорость кинка (|v| < 1)
        delta_kin (δ) – начальный сдвиг
    """
    def __init__(self, a=1.0, tau=0.0, m=1.0, u_kin=0.5, delta_kin=0.0):
        super().__init__(a, tau)
        self.m = m
        self.u_kin = u_kin          # скорость кинка
        self.delta_kin = delta_kin  # сдвиг

        self.gamma = 1.0 / np.sqrt(1.0 - self.u_kin**2)

    def exact_solution(self, x, t):
        """u(x,t) = 4·arctan( exp( m·γ·(x - v·t) + δ ) ),   γ = 1/√(1 - v²)"""
        arg = self.m * self.gamma * (x - self.u_kin * t) + self.delta_kin
        return 4.0 * np.arctan(np.exp(arg))

    def initial_derivative(self, x, t):
        arg = self.m * self.gamma * (t - self.u_kin*x) + self.delta_kin
        exp_arg = np.exp(arg)
        return -4.0 * self.m * self.gamma * self.u_kin * exp_arg / (1.0 + exp_arg**2)

    def source_term(self, u_current, u_delayed, x=None, t=None):
        """f(u) = -sin(u)"""
        return -np.sin(u_delayed)

    def initial_condition(self, x, t):
        return self.exact_solution(x, t)

    @classmethod
    def param_info(cls):
        return [
            ('a', 1.0, 'Скорость распространения возмущений'),
            ('tau', 0.0, 'Постоянное запаздывание (τ)'),
            ('m', 1.0, 'Параметр формы кинка'),
            ('u_kin', 0.5, 'Скорость кинка (|v| < 1)'),
            ('delta_kin', 0.0, 'Сдвиг кинка (δ)'),
        ]

    @classmethod
    def description(cls):
        return ("Уравнение:  u_tt = a²*u_xx - sin(u)\n"
                "Аналитический кинк: u(x,t) = 4*arctan(exp(m*gamma*(x - u_kin*t) + delta_kin))\n"
                "gamma = 1/sqrt(1 - u_kin^2)")

