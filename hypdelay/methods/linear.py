import numpy as np
from hypdelay.core.base import PDEMethod

class PWLinearInterpExtrap(PDEMethod):
    """
    кусночно-линейная экстраполяция для u_current
    кусночно-линейная интерполяция для u_delayed.
    """
    def get_current_extrapolation(self, cfg, u_prev, u_prev2):
        # Линейная экстраполяция
        delta = cfg.delta
        s = cfg.s
        return ((delta + s) * u_prev - s * u_prev2) / delta

    def get_delayed_value(self, cfg, history, A, t_current, x_idx):
        tau = cfg.tau
        t0 = cfg.t0
        delta = cfg.delta
        s = cfg.s
        t_delayed = t_current - tau

        if t_delayed < t0:
            # Получаем значения для интерполяции
            t_prev, t_prev2, u_prev, u_prev2 = history.get_interval(t_delayed, x_idx)

            # Кусочно-линейная интерполяция
            u_delayed = 1/delta * ((t_prev2 - t_current - s) * u_prev + (t_current + s - t_prev) * u_prev2)
            return u_delayed

        else:
            # Интерполяция по уже вычисленным слоям (A)
            # A – матрица существующих решений, где индекс 0 соответствует t = t0

            idx = (t_delayed - t0) / delta
            k = int(np.floor(idx))

            # Корректировка границ
            if k < 0:
                k = 0
            if k >= len(A) - 1:
                k = len(A) - 2

            t_prev2 = t0 + k * delta
            t_prev = t0 + (k + 1) * delta
            u_prev2 = A[k][x_idx]
            u_prev = A[k + 1][x_idx]

            # Кусочно-линейная интерполяция
            u_delayed = 1 / delta * ((t_prev2 - t_current - s) * u_prev + (t_current + s - t_prev) * u_prev2)
            return u_delayed