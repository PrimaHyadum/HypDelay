from hypdelay.core.base import PDEModel
import numpy as np

class DoubleSineGordonKinkModel(PDEModel):
    """
    Двойное уравнение синус-Гордона:
        u_tt - a² u_xx + sin(u) + λ sin(2u) = 0

    Аналитическое решение – кинк:
        u(x,t) = 2·arctan( √(1/(1+2λ)) · tanh( √(1+2λ)/2 · m·γ·(x - v·t) + δ ) ),
        γ = 1/√(1 - v²)

    Параметры:
        m         – параметр формы (обычно 1)
        u_kin (v) – скорость кинка (|v| < 1)
        delta_kin (δ) – начальный сдвиг
        lam (λ)   – коэффициент при sin(2u)
    """
    def __init__(self, a=1.0, tau=0.0, m=1.0, u_kin=0.5, delta_kin=0.0, lam=0.5):
        super().__init__(a, tau)
        self.m = m
        self.u_kin = u_kin
        self.delta_kin = delta_kin
        self.lam = lam

        # Лоренц-фактор
        self.gamma = 1.0 / np.sqrt(1.0 - self.u_kin**2)
        # Коэффициент, связанный с λ
        self.k = np.sqrt(1 + 2 * self.lam)
        # Амплитуда перед tanh
        self.A = np.sqrt(1 / (1 + 2 * self.lam))
        # Коэффициент перед аргументом tanh
        self.c = self.k / 2 * self.m * self.gamma

    def exact_solution(self, x, t):
        """
        u(x,t) = 2·arctan( A·tanh( c·(x - v·t) + δ ) )
        """
        arg = self.c * (x - self.u_kin * t) + self.delta_kin
        return 2.0 * np.arctan(self.A * np.tanh(arg))

    def initial_condition(self, x, t):
        return self.exact_solution(x, t)

    def initial_derivative(self, x, t):
        """
        Производная по времени в момент t.
        Вывод: du/dt = - 2·A·c·v·sech²(θ) / (1 + (A·tanh(θ))²),
        где θ = c·(x - v·t) + δ.
        """
        theta = self.c * (x - self.u_kin * t) + self.delta_kin
        sech2 = 1.0 / np.cosh(theta)**2   # sech²(θ)
        denom = 1.0 + (self.A * np.tanh(theta))**2
        return -2.0 * self.A * self.c * self.u_kin * sech2 / denom

    def source_term(self, u_current, u_delayed, x=None, t=None):
        """
        Правая часть: f = - sin(u) - λ sin(2u).
        Здесь u_current – значение в момент t (экстраполированное),
        u_delayed – значение с запаздыванием (используется для нелинейности).
        В классическом двойном синус-Гордоне запаздывание обычно не вводится,
        поэтому оба аргумента можно использовать по желанию.
        """
        return -np.sin(u_current) - self.lam * np.sin(2 * u_delayed)

    @classmethod
    def param_info(cls):
        return [
            ('a', 1.0, 'Скорость распространения возмущений'),
            ('tau', 0.0, 'Постоянное запаздывание (τ)'),
            ('m', 1.0, 'Параметр формы кинка'),
            ('u_kin', 0.5, 'Скорость кинка (|v| < 1)'),
            ('delta_kin', 0.0, 'Сдвиг кинка (δ)'),
            ('lam', 0.5, 'Коэффициент λ при sin(2u)'),
        ]

    @classmethod
    def description(cls):
        return ("Уравнение:  u_tt = a^2 * u_xx - sin(u(x, t)) - lambda * sin(2 * u(x, t - tau))\n"
                "Аналитический кинк: u(x,t) = 2·arctan( √(1/(1+2λ))·tanh( √(1+2λ)/2 · m·γ·(x - v·t) + δ ) )\n"
                "γ = 1/√(1 - v²)")