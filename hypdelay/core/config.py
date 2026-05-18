from hypdelay.core.base import PDEMethod, PDEModel

class Config:
    """Хранит все параметры задачи и схемы, а также вычисляемые величины."""
    def __init__(self, model: PDEModel, method: PDEMethod, x0: float, t0: float,
                 s: float, L: float, T: float, N: int, M: int):
        self.model = model
        self.method = method
        self.x0 = x0
        self.t0 = t0
        self.s = s
        self.L = L
        self.T = T
        self.N = N
        self.M = M

        # Параметры модели
        self.a = self.model.a
        self.tau = self.model.tau

        # Вычисляемые параметры сетки
        self.h = (L - x0) / (N - 1)
        self.delta = (T - t0) / (M - 1)
        self.eps = 1e-12

        # Вспомогательные коэффициенты для схемы
        self.rh = (self.a * self.delta / self.h) ** 2
        self.r = self.s * self.rh