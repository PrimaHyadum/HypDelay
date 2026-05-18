from hypdelay.core.base import PDEModel
import numpy as np

class Phi4KinkModel(PDEModel):
    """
    Уравнение φ⁴: u_tt - a² u_xx + m² (u³ - u) = 0

    Аналитическое решение – кинк:
        u(x,t) = tanh( m/√2 * γ * (x - v·t) + δ ),   γ = 1/√(1 - v²)

    Параметры:
        m         – параметр формы (обычно 1)
        u_kin (v) – скорость кинка (|v| < 1)
        delta_kin (δ) – начальный сдвиг
    """
    def __init__(self, a=1.0, tau=0.0, m=1.0, u_kin=0.5, delta_kin=0.0):
        super().__init__(a, tau)
        self.m = m
        self.u_kin = u_kin
        self.delta_kin = delta_kin

        # Лоренц-фактор
        self.gamma = 1.0 / np.sqrt(1.0 - self.u_kin**2)
        # Коэффициент перед аргументом tanh
        self.c = self.m / np.sqrt(2) * self.gamma

    def exact_solution(self, x, t):
        """u(x,t) = tanh( c·(x - v·t) + δ )"""
        arg = self.c * (x - self.u_kin * t) + self.delta_kin
        return np.tanh(arg)

    def initial_condition(self, x, t):
        return self.exact_solution(x, t)

    def initial_derivative(self, x, t):
        """
        Производная по времени в момент t.
        ∂u/∂t = - c * u_kin * sech²( c·(x - v·t) + δ )
        """
        arg = self.c * (x - self.u_kin * t) + self.delta_kin
        sech = 1.0 / np.cosh(arg)
        return -self.c * self.u_kin * sech * sech

    def source_term(self, u_current, u_delayed, x=None, t=None):
        """Правая часть: f = m² (u - u³) (запаздывание по второму аргументу)"""
        return self.m**2 * (u_current - u_delayed**3)

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
        return ("Уравнение:  u_tt = a^2 * u_xx - m^2(u^3(x, t) - u(x, t - tau)\n"
                "Аналитический кинк: u(x,t) = tanh(m/sqrt(2)*gamma*(x - u_kin*t) + delta_kin)\n"
                "gamma = 1/sqrt(1 - u_kin^2)")