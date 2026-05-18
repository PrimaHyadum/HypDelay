import numpy as np
import matplotlib.pyplot as plt
from hypdelay.core.config import Config
from hypdelay.core.base import PDEModel

def plot_solution(model: PDEModel, A: np.ndarray, cfg: Config, title: str = "Решение", show: bool =True):
    """
    Отображает трёхмерный график численного решения (красные точки)
    и каркас точного решения (зелёная сетка).

    :param show: bool
    :param model: модель PDE (для точного решения)
    :param A: численное решение (M x N)
    :param cfg: конфигурация (содержит x0, L, t0, T, N, M, h, delta)
    :param title: заголовок окна
    """
    M, N = cfg.M, cfg.N
    h, delta = cfg.h, cfg.delta
    x0, L = cfg.x0, cfg.L
    t0, T = cfg.t0, cfg.T

    # Сбор координат для численного решения
    XX, TT, ZZ = [], [], []
    for i in range(M):
        t_i = i * delta + t0
        for j in range(N):
            x_j = j * h + x0
            XX.append(x_j)
            TT.append(t_i)
            ZZ.append(A[i, j])

    XX, TT, ZZ = np.array(XX), np.array(TT), np.array(ZZ)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(XX, TT, ZZ, color='red', s=12, alpha=1.0, edgecolors='none',
               label='Численное решение')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('φ')
    ax.set_title(title)

    # Точное решение на более частой сетке для каркаса
    xx = np.linspace(x0, L, 100)
    tt = np.linspace(t0, T, 100)
    X_grid, T_grid = np.meshgrid(xx, tt)
    Z_exact = model.exact_solution(X_grid, T_grid)
    ax.plot_wireframe(X_grid, T_grid, Z_exact, color='green', alpha=0.6,
                      linewidth=0.5, label='Точное решение')
    ax.legend()

    if show:
        plt.show()
    else:
        return plt.gcf()