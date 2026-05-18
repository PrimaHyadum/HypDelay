import numpy as np
from hypdelay.core.config import Config

class History:
    """
    Хранит решение u(x, t) для t ∈ [t0 - tau, t0] с постоянным шагом delta.

    :params:
    :param A_t0: Первый слой матрицы A, который соответствует решению u(x, t0), x in [0, L]
    """
    def __init__(self, cfg: Config, A_t0):
        self.cfg = cfg
        self.tau = cfg.tau
        self.t0 = cfg.t0
        self.x0 = cfg.x0
        self.delta = cfg.delta
        self.h = cfg.h
        self.N_his = cfg.N
        self.s = cfg.s
        self.eps = cfg.eps
        self.model = cfg.model

        self.M_his = int(round(self.tau / self.delta)) + 1
        self.values = np.zeros((self.M_his, self.N_his))
        self.values[0] = A_t0.copy()

    def build_from_model(self):
        """
        Заполняет историю, вызывая model.initial_condition(x, t)
        для всех t = -i*delta, i in [1, M_his]
        """

        for i in range(1, self.M_his): # values[0] = A_t0
            t_i = self.t0 - i * self.delta
            for j in range(self.N_his):
                x_j = j * self.h + self.x0
                self.values[i, j] = self.model.initial_condition(x_j, t_i)

    def get_interval(self, t_target, x_idx):
        """
        Возвращает кортеж (t_left, t_right, u_left, u_right) для заданного t_target.
        t_left и t_right – узлы времени, между которыми лежит t_target.
        u_left, u_right – значения решения в этих узлах для позиции x_idx.
        """
        t_min = self.t0 - self.tau
        if not (t_min <= t_target <= self.t0):
            raise ValueError(f"t_target {t_target} вне [{t_min}, {self.t0}]")

        # Вычисляем индекс слоя слева от t_target
        # Поскольку values[0] = t0, values[i] = t0 - i*delta
        idx = (self.t0 - t_target) / self.delta

        # Если idx очень близок к целому, округляем его
        # Используем относительный допуск 1e-12
        if abs(idx - round(idx)) < self.eps:
            k = int(round(idx))
        else:
            k = int(np.floor(idx))

        # Корректировка границ
        if k < 0:
            k = 0
        if k >= self.M_his - 1:
            k = self.M_his - 2

        # Узлы: более ранний (k+1) и более поздний (k)
        t_prev = self.t0 - k * self.delta  # более поздний момент (больше)
        t_prev2 = self.t0 - (k + 1) * self.delta  # более ранний момент (меньше)
        u_prev = self.values[k, x_idx]
        u_prev2 = self.values[k + 1, x_idx]

        return t_prev, t_prev2, u_prev, u_prev2