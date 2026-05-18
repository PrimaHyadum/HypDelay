import numpy as np
from hypdelay.core.config import Config

def compute_error(A: np.ndarray, cfg: Config):
    """
    Вычисляет максимальную по модулю и среднеквадратичную ошибку
    относительно точного решения модели.

    :param A: численное решение (M x N)
    :param cfg: конфигурация (содержит x0, L, t0, T, N, M)
    :return: (max_err, l2_err, mae)
    """
    model = cfg.model
    M, N = cfg.M, cfg.N
    x = np.linspace(cfg.x0, cfg.L, N)
    t = np.linspace(cfg.t0, cfg.T, M)
    X, T_grid = np.meshgrid(x, t)
    exact = model.exact_solution(X, T_grid)

    diff = np.abs(A - exact)
    max_err = np.max(diff)
    l2_err = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(A - exact))
    return max_err, l2_err, mae