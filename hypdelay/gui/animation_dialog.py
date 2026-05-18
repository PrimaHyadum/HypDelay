import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from hypdelay.utils.plot_animation import PlotAnimation

class AnimationDialog(tk.Toplevel):
    """Окно настройки и просмотра 3D-анимации решений."""
    def __init__(self, parent, solutions, errors=None):
        super().__init__(parent)
        self.title("Анимация решений (3D)")
        self.geometry("1000x750")
        self.solutions = solutions
        self.plot_anim = None
        self.ani = None
        self.canvas = None
        self.is_running = False
        self.errors = errors

        # Панель управления
        control_frame = ttk.Frame(self, padding=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # FPS
        ttk.Label(control_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="30")
        fps_entry = ttk.Entry(control_frame, textvariable=self.fps_var, width=5)
        fps_entry.pack(side=tk.LEFT, padx=5)

        # Тип графика
        ttk.Label(control_frame, text="Тип:").pack(side=tk.LEFT, padx=(10,0))
        self.plot_type_var = tk.StringVar(value="wireframe")
        type_combo = ttk.Combobox(control_frame, textvariable=self.plot_type_var,
                                  values=["surface", "wireframe"], state="readonly", width=10)
        type_combo.pack(side=tk.LEFT, padx=5)

        # Углы обзора
        ttk.Label(control_frame, text="Elev:").pack(side=tk.LEFT, padx=(10,0))
        self.elev_var = tk.StringVar(value="30")
        ttk.Entry(control_frame, textvariable=self.elev_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(control_frame, text="Azim:").pack(side=tk.LEFT, padx=(5,0))
        self.azim_var = tk.StringVar(value="-160")
        ttk.Entry(control_frame, textvariable=self.azim_var, width=4).pack(side=tk.LEFT, padx=2)

        # Зацикливание
        self.repeat_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Повтор", variable=self.repeat_var).pack(side=tk.LEFT, padx=10)

        # Кнопки
        ttk.Button(control_frame, text="Запустить", command=self.start_animation).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Остановить", command=self.stop_animation).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Сохранить", command=self.save_animation).pack(side=tk.LEFT, padx=5)

        # Фрейм для графика
        self.fig_frame = ttk.Frame(self)
        self.fig_frame.pack(fill=tk.BOTH, expand=True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_animation(self):
        if self.is_running:
            return
        try:
            fps = float(self.fps_var.get())
            elev = float(self.elev_var.get())
            azim = float(self.azim_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "FPS, Elev и Azim должны быть числами")
            return

        plot_type = self.plot_type_var.get()
        repeat = self.repeat_var.get()

        # Останавливаем предыдущую, если была
        self.stop_animation()

        self.plot_anim = PlotAnimation(self.solutions, fps=fps,
                                       plot_type=plot_type, elev=elev, azim=azim,
                                       repeat=repeat, errors=self.errors)
        self.plot_anim.create_animation()
        self.ani = self.plot_anim.ani

        # Встраиваем холст
        self.canvas = FigureCanvasTkAgg(self.plot_anim.fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Запускаем таймер анимации
        self.ani.event_source.start()
        self.is_running = True

    def stop_animation(self):
        if self.ani and self.is_running:
            self.ani.event_source.stop()
            self.is_running = False
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.canvas = None

    def save_animation(self):
        if self.ani is None:
            messagebox.showinfo("Нет анимации", "Сначала запустите анимацию.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".gif",
                                            filetypes=[("GIF files", "*.gif"), ("MP4 files", "*.mp4")])
        if not path:
            return
        writer = 'imagemagick' if path.endswith('.gif') else 'ffmpeg'
        try:
            self.plot_anim.save(path, writer=writer)
            messagebox.showinfo("Сохранено", f"Анимация сохранена в {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def on_close(self):
        self.stop_animation()
        self.destroy()