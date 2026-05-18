import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import numpy as np
import csv

from hypdelay.models import (
    SineGordonKinkModel, Phi4KinkModel, DoubleSineGordonKinkModel, HyperbolicModel
)
from hypdelay.core.solver import DelayPDESolver
from hypdelay.methods import PWLinearInterpExtrap, PrimeMethod, PrimeHybridMethod
from hypdelay.utils import compute_error, plot_solution, save_all_results
from hypdelay.gui.dialogs import CSVLoaderDialog, SweepWindow
from hypdelay.gui.tooltip import ToolTip

DEFAULT_S = 0.3
DEFAULT_L = 10.0
DEFAULT_T = 5.0
DEFAULT_N = 50
DEFAULT_M = 50

METHODS = {
    "Кусочно-линейная интерполяция": PWLinearInterpExtrap(),
    "Прямой": PrimeMethod(),
    "Гибридный": PrimeHybridMethod(),
}

class ModelParamsFrame(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Параметры модели")
        self.entries = {}
        self.columnconfigure(1, weight=1)

    def configure(self, model_cls):
        for widget in self.winfo_children():
            widget.destroy()
        self.entries.clear()
        for i, (name, default, tip) in enumerate(model_cls.param_info()):
            lbl = ttk.Label(self, text=name)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=2, pady=2)
            ToolTip(lbl, tip)
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(self, textvariable=var, width=12)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=2, pady=2)
            self.entries[name] = var

    def get_values(self):
        return {name: float(var.get()) for name, var in self.entries.items()}

    def set_values(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.entries:
                self.entries[k].set(str(v))

class SchemeParamsFrame(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Параметры схемы")
        self.entries = {}
        fields = [
            ('method', 'Кусочно-линейная интерполяция', 'Численный метод интерполяции/экстраполяции'),
            ('s', str(DEFAULT_S), 'Весовой коэффициент (0 – явная, 1 – неявная)'),
            ('L', str(DEFAULT_L), 'Длина отрезка по x'),
            ('T', str(DEFAULT_T), 'Конечное время'),
            ('N', str(DEFAULT_N), 'Число узлов по пространству'),
            ('M', str(DEFAULT_M), 'Число узлов по времени (обязательно)')
        ]
        for i, (name, default, tip) in enumerate(fields):
            lbl = ttk.Label(self, text=name)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=2, pady=2)
            ToolTip(lbl, tip)
            if name == 'method':
                var = tk.StringVar(value=default)
                cb = ttk.Combobox(self, textvariable=var, values=list(METHODS.keys()), state="readonly", width=25)
                cb.grid(row=i, column=1, sticky=tk.EW, padx=2, pady=2)
                self.entries[name] = var
            else:
                var = tk.StringVar(value=default)
                entry = ttk.Entry(self, textvariable=var, width=12)
                entry.grid(row=i, column=1, sticky=tk.EW, padx=2, pady=2)
                self.entries[name] = var
        self.columnconfigure(1, weight=1)

    def get_values(self):
        vals = {
            'method': self.entries['method'].get(),
            's': float(self.entries['s'].get()),
            'L': float(self.entries['L'].get()),
            'T': float(self.entries['T'].get()),
            'N': int(self.entries['N'].get()),
        }
        M_str = self.entries['M'].get().strip()
        if not M_str:
            raise ValueError("Параметр M обязателен для заполнения.")
        try:
            vals['M'] = int(M_str)
        except ValueError:
            raise ValueError("M должно быть целым числом.")
        return vals

    def set_values(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.entries:
                self.entries[k].set(str(v))

class PlotWindow(tk.Toplevel):
    def __init__(self, parent, A, cfg, model, model_args, scheme_args, max_err, l2_err, mae, on_close_callback=None):
        super().__init__(parent)
        self.title("Визуализация решения")
        self.geometry("1100x700")

        self.on_close_callback = on_close_callback

        self.model = model
        self.cfg = cfg
        self.A = A
        self.model_args = model_args
        self.scheme_args = scheme_args
        self.max_err = max_err
        self.l2_err = l2_err
        self.mae = mae

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Настройка сетки
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(0, weight=1)

        # Фрейм для графика
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=0, sticky="nsew")

        self.fig = plot_solution(model, A, cfg, title="Численное решение", show=False)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Правая информационная панель
        info_frame = ttk.Frame(main_frame, width=370)
        info_frame.grid(row=0, column=1, sticky="ns", padx=(10, 10), pady=10)
        info_frame.grid_propagate(False)  # сохраняем ширину 370 px

        ttk.Label(info_frame, text="Параметры модели", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        model_text = "\n".join([f"{k} = {v}" for k, v in self.model_args.items()])
        ttk.Label(info_frame, text=model_text, justify=tk.LEFT, wraplength=350).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(info_frame, text="Параметры схемы", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        scheme_show = {k: v for k, v in self.scheme_args.items()}
        scheme_show['N'] = cfg.N
        scheme_show['M'] = cfg.M
        scheme_show['method'] = self.scheme_args.get('method', 'Linear')
        scheme_text = "\n".join([f"{k} = {v}" for k, v in scheme_show.items()])
        ttk.Label(info_frame, text=scheme_text, justify=tk.LEFT, wraplength=350).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(info_frame, text="Сетка", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        dh_ratio = cfg.delta / cfg.h if cfg.h != 0 else 0
        MN_ratio = cfg.M / cfg.N if cfg.N != 0 else 0
        grid_text = (f"h = {cfg.h:.4f}\nΔ = {cfg.delta:.4f}\n"
                     f"Δ/h = {dh_ratio:.4f}\nM/N = {MN_ratio:.4f}")
        ttk.Label(info_frame, text=grid_text, justify=tk.LEFT, wraplength=350).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(info_frame, text="Погрешность", font=('', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        err_text = (f"Максимальная ошибка = {self.max_err:.6g}\n"
                    f"L2 ошибка = {self.l2_err:.6g}\n"
                    f"MAE = {self.mae:.6g}")
        ttk.Label(info_frame, text=err_text, justify=tk.LEFT, wraplength=350).pack(anchor=tk.W, pady=(0, 10))

        # Кнопки вертикально
        btn_frame = ttk.Frame(info_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_frame, text="Сохранить график (PNG)", command=self.save_figure).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Сохранить координаты (CSV)", command=self.save_solution_csv).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Сохранить параметры (CSV)", command=self.save_params_csv).pack(fill=tk.X, pady=2)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        if self.on_close_callback:
            self.on_close_callback()
        self.destroy()

    def save_figure(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
        )
        if filepath:
            self.fig.savefig(filepath, dpi=150)
            messagebox.showinfo("Сохранено", f"График сохранён как {filepath}")

    def save_params_csv(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filepath:
            return
        extra = {'max_err': self.max_err, 'l2_err': self.l2_err, 'mae': self.mae}
        save_all_results([(self.cfg, extra)], filepath)
        messagebox.showinfo("Готово", f"Параметры сохранены в {filepath}")

    def save_solution_csv(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filepath:
            return
        x = np.linspace(self.cfg.x0, self.cfg.L, self.cfg.N)
        t = np.linspace(self.cfg.t0, self.cfg.T, self.cfg.M)
        X, T_grid = np.meshgrid(x, t)
        Z = self.A
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 't', 'u'])
            for i in range(self.cfg.M):
                for j in range(self.cfg.N):
                    writer.writerow([X[i, j], T_grid[i, j], Z[i, j]])
        messagebox.showinfo("Готово", f"Координаты сохранены в {filepath}")


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HypDelay")
        self.geometry("900x700")

        self.models = {
            "Sine-Gordon (кинк)": SineGordonKinkModel,
            "Phi-4 (кинк)": Phi4KinkModel,
            "Double Sine-Gordon": DoubleSineGordonKinkModel,
            "Hyperbolic test (with delay)": HyperbolicModel,
        }
        self.models_by_class = {cls.__name__: cls for cls in self.models.values()}
        self.current_model_cls = SineGordonKinkModel
        self.current_model_obj = None

        menubar = tk.Menu(self)
        self.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Загрузить CSV...", command=self.load_csv)
        file_menu.add_command(label="Сохранить результат в CSV", command=self.save_single_result)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        top_frame = ttk.Frame(self, padding=5)
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="Модель:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=list(self.models.keys())[0])
        model_cb = ttk.Combobox(top_frame, textvariable=self.model_var,
                                values=list(self.models.keys()), state="readonly")
        model_cb.pack(side=tk.LEFT, padx=5)
        model_cb.bind("<<ComboboxSelected>>", self.on_model_change)

        ttk.Button(top_frame, text="Загрузить CSV", command=self.load_csv).pack(side=tk.RIGHT, padx=5)

        eq_frame = ttk.LabelFrame(self, text="Уравнение и решение", padding=5)
        eq_frame.pack(fill=tk.X, padx=5, pady=5)
        eq_text = self.current_model_cls.description()
        self.eq_label = ttk.Label(eq_frame, text=eq_text, font=('', 9))
        self.eq_label.pack(anchor=tk.W)

        if os.path.exists("formula.png"):
            try:
                img = tk.PhotoImage(file="formula.png")
                img_label = ttk.Label(eq_frame, image=img)
                img_label.image = img
                img_label.pack(anchor=tk.CENTER, pady=5)
            except Exception:
                pass

        param_frame = ttk.Frame(self, padding=5)
        param_frame.pack(fill=tk.BOTH, expand=True)

        self.model_params = ModelParamsFrame(param_frame)
        self.model_params.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.scheme_params = SchemeParamsFrame(param_frame)
        self.scheme_params.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.model_params.configure(self.current_model_cls)

        bottom_frame = ttk.Frame(self, padding=5)
        bottom_frame.pack(fill=tk.X)
        ttk.Button(bottom_frame, text="Решить", command=self.solve).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Перебор параметров...", command=self.open_sweep).pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(bottom_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        self.results_storage = []
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите закрыть программу?"):
            self.destroy()

    def on_model_change(self, event=None):
        name = self.model_var.get()
        self.current_model_cls = self.models[name]
        self.model_params.configure(self.current_model_cls)
        if hasattr(self.current_model_cls, 'description'):
            self.eq_label.config(text=self.current_model_cls.description())

    def load_csv(self):
        filepath = filedialog.askopenfilename(
            title="Выберите CSV с параметрами",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filepath:
            return
        dialog = CSVLoaderDialog(self, filepath)
        self.wait_window(dialog)
        if dialog.result:
            res = dialog.result
            model_name = res.get('model', None)
            if model_name:
                for name, cls in self.models.items():
                    if cls.__name__ == model_name:
                        if self.current_model_cls != cls:
                            self.model_var.set(name)
                            self.on_model_change()
                        break
            else:
                self.model_params.configure(self.current_model_cls)

            param_info = self.current_model_cls.param_info()
            model_dict = {}
            for name, default, _ in param_info:
                val_str = res.get(name, str(default))
                try:
                    val = float(val_str)
                except (ValueError, TypeError):
                    val = default
                model_dict[name] = val
            self.model_params.set_values(**model_dict)

            try:
                N_val = int(float(res.get('N', str(DEFAULT_N))))
            except (ValueError, TypeError):
                N_val = DEFAULT_N

            M_str = str(res.get('M', '')).strip()
            M_val = None
            if M_str:
                try:
                    M_val = int(float(M_str))
                except (ValueError, TypeError):
                    pass

            method = res.get('method', 'Кусочно-линейная интерполяция')
            self.scheme_params.set_values(
                method=method,
                s=float(res.get('s', DEFAULT_S)),
                L=float(res.get('L', DEFAULT_L)),
                T=float(res.get('T', DEFAULT_T)),
                N=N_val
            )
            if M_val is not None:
                self.scheme_params.entries['M'].set(str(M_val))
            else:
                self.scheme_params.entries['M'].set('')

    def save_single_result(self):
        if not self.results_storage:
            messagebox.showinfo("Нет данных", "Сначала выполните расчёт.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".csv")
        if filepath:
            save_all_results(self.results_storage, filepath)

    def solve(self):
        try:
            model_args = self.model_params.get_values()
            scheme_args = self.scheme_params.get_values()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Некорректные данные:\n{e}")
            return

        model = self.current_model_cls(**model_args)
        self.current_model_obj = model

        method_name = scheme_args['method']
        method = METHODS.get(method_name, PWLinearInterpExtrap())
        s = scheme_args['s']
        L = scheme_args['L']
        T = scheme_args['T']
        N = scheme_args['N']
        M = scheme_args['M']          # теперь M обязателен, вычислять не нужно
        x0 = 0.0
        t0 = 0.0

        solver = DelayPDESolver(model, method, x0=x0, t0=t0, s=s, L=L, T=T, N=N, M=M)
        self.progress.start()
        threading.Thread(target=self._solve_thread, args=(solver, model, model_args, scheme_args), daemon=True).start()

    def _solve_thread(self, solver, model, model_args, scheme_args):
        try:
            A, cfg = solver.solve(verbose=False)
            max_err, l2_err, mae = compute_error(A, cfg)
            # Сохраняем результат в виде (cfg, extra)
            extra = {'max_err': max_err, 'l2_err': l2_err, 'mae': mae}
            self.results_storage.append((cfg, extra))
            self.after(0, self._on_solve_success, A, cfg, model, model_args, scheme_args, max_err, l2_err, mae)
        except Exception as e:
            self.after(0, messagebox.showerror, "Ошибка расчёта", str(e))
        finally:
            self.after(0, self.progress.stop)

    def _on_solve_success(self, A, cfg, model, model_args, scheme_args, max_err, l2_err, mae):
        PlotWindow(self, A, cfg, model, model_args, scheme_args, max_err, l2_err, mae)

    def open_sweep(self):
        try:
            model_args = self.model_params.get_values()
            scheme_args = self.scheme_params.get_values()
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные параметры: {e}")
            return
        SweepWindow(self, self.models, self.current_model_cls, model_args, scheme_args)

    def open_plot_window(self, A, cfg, model, model_args, scheme_args, max_err, l2_err, mae, on_close_callback=None):
        PlotWindow(self, A, cfg, model, model_args, scheme_args, max_err, l2_err, mae, on_close_callback)

if __name__ == "__main__":
    app = Application()
    app.mainloop()