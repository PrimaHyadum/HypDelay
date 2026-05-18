from hypdelay.core.base import PDEModel
import numpy as np

class HyperbolicModel(PDEModel):
    """
    гиперболическое уравнение с постоянным запаздыванием:
    u_tt = a² u_xx + a² π² exp(-t) sin(π x) + exp(τ - 2t) sin²(π x) + u (1 - u(t-τ))
    Точное решение: u(x,t) = exp(-t) * sin(π x)
    """
    def __init__(self, a=1.0, tau=0.0):
        super().__init__(a, tau)

    def exact_solution(self, x, t):
        return np.exp(-t) * np.sin(np.pi * x)

    def initial_condition(self, x, t):
        return self.exact_solution(x, t)

    def initial_derivative(self, x, t):
        return -np.exp(-t) * np.sin(np.pi * x)

    def source_term(self, u_current, u_delayed, x=None, t=None):
        # часть правой части, не зависящая от u (используется при tau=0)
        func = self.a**2 * np.pi**2 * np.exp(-t) * np.sin(np.pi * x) + np.exp(self.tau - 2 * t) * np.sin(np.pi * x)**2
        func += u_current * (1.0 - u_delayed)
        return func

    @classmethod
    def param_info(cls):
        return [
            ('a', 1.0, 'Коэффициент'),
            ('tau', 0.0, 'Постоянное запаздывание (τ)')
        ]

    @classmethod
    def description(cls):
        return ("Уравнение:  u_tt = a²*u_xx + a²*π²*exp(-t)*sin(π*x) + "
                "exp(tau-2t)*sin²(π*x) + u(x, t)*(1 - u(x,t-tau))\n"
                "Точное решение: u(x,t) = exp(-t)*sin(π*x)")