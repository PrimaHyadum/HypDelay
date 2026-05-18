import numpy as np

from hypdelay.core.config import Config
from hypdelay.core.history import History
from hypdelay.core.base import PDEModel, PDEMethod  # для аннотации типа

class DelayPDESolver:
    """
    Решатель задачи:

        u_tt = a² * u_xx + f(x, t, u(x, t), u_t(x, ·)),   x ∈ [x0, L], t ∈ [t0, T]

    Начальные и граничные условия определяются моделью (PDEModel),
    Вычисление интерполяции и экстраполяции определяется методом (PDEMethod)

    :params:
        :param model: PDEModel
            Модель уравнения
        :param method: PDEMethod
            Метод интерполяции и экстраполяции

        :param x0: float ∈ [0,+inf]
            Точка отсчета по пространству
        :param t0: float ∈ [0,+inf]
            Точка отсчета по времени
        :param s: float ∈ [0,1]
            Весовой коэффициент схемы.
        :param L: float ∈ [0,+inf]
            Конец пространственного отрезка.
        :param T: float ∈ [0,+inf]
            Конечное время.
    """

    def __init__(self, model: PDEModel, method: PDEMethod,
                 x0= 0.0, t0= 0.0, s=0.0, L=1.0, T=1.0, N=10, M=10):
        self.model = model
        self.method = method
        self.s = s
        self.L = L
        self.T = T
        self.t0 = t0
        self.x0 = x0
        self.N = N
        self.M = M

        self._cfg = Config(self.model, self.method, self.x0, self.t0,
                           self.s, self.L, self.T, self.N, self.M)

        self.eps = self._cfg.eps

    def solve(self, verbose=False):
        """
        Выполняет численное решение на сетке из N узлов по x и M узлов по t.

        :returns:
            A : ndarray (M, N) – численное решение в узлах сетки
            h : float – шаг по пространству
            delta : float – шаг по времени
        """

        N = self.N
        M = self.M

        cfg = self._cfg

        tau = cfg.tau
        h = cfg.h
        delta = cfg.delta

        # rh = (a * delta / h) ** 2
        rh = cfg.rh
        # r = s * rh
        r = cfg.r

        A = np.zeros((M, N))

        # --- Начальный слой (t = t0) ---
        for j in range(N):
            x_j = j * h + self.x0
            A[0, j] = self.model.initial_condition(x_j, self.t0)

        # --- Граничные условия на всех временах ---
        for i in range(M):
            t_i = i * delta + self.t0
            A[i, 0] = self.model.boundary_condition_x0(self.x0, t_i)
            A[i, N-1] = self.model.boundary_condition_L(self.L, t_i)

        # --- Второй слой ---
        for j in range(1, N - 1):
            x_j = j * h + self.x0
            u0 = A[0, j]
            ut0 = self.model.initial_derivative(x_j, self.t0)

            # Метод тейлора
            A[1, j] = u0 + delta * ut0

        # --- Предыстория значений [t0 - tau, t0] ---
        hist = History(cfg=cfg, A_t0=A[0])
        if tau > self.eps:
            hist.build_from_model()

        # --- Трёхдиагональная матрица C для внутренних узлов ---
        # Уравнение: (1+2r) u_j^{k+1} - r (u_{j-1}^{k+1} + u_{j+1}^{k+1}) = B_j
        C = np.zeros((N-2, N-2))

        for i in range(N-2):
            C[i, i] = 1.0 + 2.0 * r
            if i > 0:
                C[i, i-1] = -r
            if i < N-3:
                C[i, i+1] = -r

        # --- Основной цикл по времени ---
        for i in range(2, M):
            t_i = i * delta + self.t0
            B = np.zeros(N-2)

            for j in range(1, N-1):
                x_j = j * h + self.x0

                # Здесь выполняется интерполяция и экстраполяция
                u_current = self.method.get_current_extrapolation(cfg=cfg, u_prev=A[i - 1, j], u_prev2=A[i - 2, j])

                if tau < self.eps:
                    u_delayed = u_current

                else:
                    u_delayed = self.method.get_delayed_value(cfg=cfg, history=hist, A=A[:i], t_current=t_i, x_idx=j)

                    # TEST
                    # TODO: Потом убрать
                    if u_delayed is None:
                        u_delayed = u_current

                # Здесь выполняется вычисление B[j-1]
                func_with_delay = self.model.source_term(u_current, u_delayed, x_j, t_i)

                B[j - 1] = self.s * rh * (A[i - 2, j - 1] - 2 * A[i - 2, j] + A[i - 2, j + 1]) \
                           + (1 - 2 * self.s) * rh * (A[i - 1, j - 1] - 2 * A[i - 1, j] + A[i - 1, j + 1]) \
                           + 2 * A[i - 1, j] - A[i - 2, j] \
                           + func_with_delay * delta ** 2

            # Решение системы для внутренних узлов
            u_new_inner = np.linalg.solve(C, B)
            A[i, 1:N-1] = u_new_inner

            if verbose:
                print(f"Выполнен слой для t = {t_i:4f}")

        return A, cfg