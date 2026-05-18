import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PlotAnimation:
    """
    3D-анимация поверхности u(x,t) для списка решений.
    Каждый кадр – новое решение (A, cfg).
    Параметры:
        solutions: list of tuples (A, cfg)
        fps: кадров в секунду (скорость смены решений)
        figsize, dpi: параметры фигуры
        plot_type: 'surface' или 'wireframe'
        elev, azim: начальный угол обзора 3D-осей
        repeat: зацикливать ли анимацию
    """
    def __init__(self, solutions, fps=5, figsize=(10, 7), dpi=100,
                 plot_type='surface', elev=30, azim=-60, repeat=True, errors=None):
        self.solutions = solutions
        self.fps = fps
        self.figsize = figsize
        self.dpi = dpi
        self.plot_type = plot_type
        self.elev = elev
        self.azim = azim
        self.repeat = repeat
        self.fig = None
        self.ani = None
        self.errors = errors if errors is not None else []

    def create_animation(self):
        if not self.solutions:
            raise ValueError("Список решений пуст")

        # Готовим глобальные пределы для осей на основе всех решений
        x_min, x_max = np.inf, -np.inf
        t_min, t_max = np.inf, -np.inf
        u_min, u_max = np.inf, -np.inf
        for A, cfg in self.solutions:
            x = np.linspace(cfg.x0, cfg.L, cfg.N)
            t = np.linspace(cfg.t0, cfg.T, cfg.M)
            x_min = min(x_min, x[0])
            x_max = max(x_max, x[-1])
            t_min = min(t_min, t[0])
            t_max = max(t_max, t[-1])
            # Игнорируем некорректные значения при вычислении u_min/u_max
            fin = np.isfinite(A)
            if np.any(fin):
                u_min = min(u_min, np.min(A[fin]))
                u_max = max(u_max, np.max(A[fin]))
        # На случай, если все данные плохие
        if not np.isfinite(u_min) or not np.isfinite(u_max):
            u_min, u_max = -1, 1

        # Создаём фигуру и 3D-оси
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(t_min, t_max)
        self.ax.set_zlim(u_min, u_max)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('t')
        self.ax.set_zlabel('u')
        self.ax.view_init(elev=self.elev, azim=self.azim)
        title = self.fig.suptitle('', fontsize=11)

        # Объект для поверхности (будет удаляться и создаваться заново на каждом кадре)
        self.surface = None

        def animate(idx):
            # Удаляем предыдущую поверхность (если была)
            if self.surface:
                self.surface.remove()
                self.surface = None

            A, cfg = self.solutions[idx]
            x = np.linspace(cfg.x0, cfg.L, cfg.N)
            t = np.linspace(cfg.t0, cfg.T, cfg.M)
            T_grid, X_grid = np.meshgrid(t, x, indexing='ij')  # форма (M, N)
            Z = A  # A уже имеет форму (M, N)

            # Убираем бесконечности/NaN для безопасного отображения
            Z_plot = np.where(np.isfinite(Z), Z, np.nan)

            if self.plot_type == 'wireframe':
                self.surface = self.ax.plot_wireframe(X_grid, T_grid, Z_plot,
                                                      rstride=1, cstride=1,
                                                      color='darkblue', linewidth=0.5)
            else:  # surface
                self.surface = self.ax.plot_surface(X_grid, T_grid, Z_plot,
                                                    cmap='viridis', edgecolor='none',
                                                    antialiased=True, alpha=0.9)

            # Обновляем заголовок с параметрами
            model_params = []
            if hasattr(cfg.model, 'a'):
                model_params.append(f"a={cfg.model.a}")
            if hasattr(cfg.model, 'tau'):
                model_params.append(f"tau={cfg.model.tau}")
            model_str = ", ".join(model_params)
            info = (f"Решение {idx+1}/{len(self.solutions)} | {model_str}\n"
                    f"N={cfg.N}, M={cfg.M}, s={cfg.s:.3f}")
            if idx < len(self.errors):
                max_e, l2_e, mae_e = self.errors[idx]
                info += f"\nМаксимальная ошибка={max_e:.4g}  L2={l2_e:.4g}  MAE={mae_e:.4g}"
            title.set_text(info)
            return [self.surface, title]

        self.ani = FuncAnimation(self.fig, animate, frames=len(self.solutions),
                                 interval=1000/self.fps, blit=False, repeat=self.repeat)
        return self.ani

    def show(self):
        if self.ani is None:
            self.create_animation()
        plt.show()

    def save(self, path, writer='imagemagick'):
        if self.ani is None:
            self.create_animation()
        self.ani.save(path, writer=writer, fps=self.fps)