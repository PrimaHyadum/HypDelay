import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import csv
import numpy as np

from hypdelay.core.solver import DelayPDESolver
from hypdelay.methods import PWLinearInterpExtrap, PrimeMethod, PrimeHybridMethod
from hypdelay.utils import compute_error
from hypdelay.utils import check_stability   # убедитесь, что модуль существует
from hypdelay.gui.animation_dialog import AnimationDialog

# Константы по умолчанию
DEFAULT_S = 0.3
DEFAULT_L = 10.0
DEFAULT_T = 5.0
DEFAULT_N = 50
DEFAULT_M = 50

# Словарь доступных методов
METHODS = {
    "Кусочно-линейная интерполяция": PWLinearInterpExtrap(),
    "Прямой": PrimeMethod(),
    "Гибридный": PrimeHybridMethod(),
}

# Список моделей (будет передан из главного окна)
ALL_MODELS = {}   # заполняется в main_window при создании SweepWindow


class CSVLoaderDialog(tk.Toplevel):
    """Окно выбора строки из CSV-файла с параметрами."""
    def __init__(self, parent, filepath):
        super().__init__(parent)
        self.title("Выберите строку с параметрами")
        self.geometry("800x400")
        self.result = None

        try:
            with open(filepath, 'r', newline='') as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать CSV:\n{e}")
            self.destroy()
            return

        if not self.data:
            messagebox.showinfo("Пусто", "Файл не содержит данных.")
            self.destroy()
            return

        columns = list(self.data[0].keys())
        tree = ttk.Treeview(self, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        for row in self.data:
            tree.insert('', tk.END, values=[row[c] for c in columns])

        tree.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Выбрать", command=lambda: self.on_select(tree)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side=tk.LEFT, padx=5)

    def on_select(self, tree):
        selection = tree.selection()
        if not selection:
            messagebox.showwarning("Не выбрано", "Выберите строку.")
            return
        item = tree.item(selection[0])['values']
        columns = list(self.data[0].keys())
        self.result = dict(zip(columns, item))
        self.destroy()


class SweepWindow(tk.Toplevel):
    """Окно параметрического перебора с фильтрами."""
    def __init__(self, parent, models_dict, model_cls, base_model_args, base_scheme_args):
        super().__init__(parent)
        self.title("Перебор параметров")
        self.geometry("1200x850")
        self.models_dict = models_dict
        self.model_cls = model_cls
        self.solutions_data = []
        self.solutions_errors = []

        # Исключаем method из перебираемых параметров
        self.base_scheme_args = {k: v for k, v in base_scheme_args.items() if k != 'method'}
        self.current_method = base_scheme_args.get('method', 'Кусочно-линейная интерполяция')

        if 'M' not in self.base_scheme_args or self.base_scheme_args['M'] is None:
            self.base_scheme_args['M'] = DEFAULT_M

        # ------------------ Верхняя панель: модель и метод ------------------
        top_frame = ttk.Frame(self, padding=5)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Модель:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=self._model_name_by_cls(model_cls))
        model_cb = ttk.Combobox(top_frame, textvariable=self.model_var,
                                values=list(self.models_dict.keys()), state="readonly", width=30)
        model_cb.pack(side=tk.LEFT, padx=5)
        model_cb.bind("<<ComboboxSelected>>", self.on_model_change)

        ttk.Label(top_frame, text="Метод:").pack(side=tk.LEFT, padx=(20, 0))
        self.method_var = tk.StringVar(value=self.current_method)
        method_cb = ttk.Combobox(top_frame, textvariable=self.method_var,
                                 values=list(METHODS.keys()), state="readonly", width=25)
        method_cb.pack(side=tk.LEFT, padx=5)

        # ------------------ Основная область: параметры + фильтр ------------------
        main_panel = ttk.Frame(self)
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Левая часть – прокручиваемый список параметров
        left_frame = ttk.Frame(main_panel)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(left_frame)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Правая часть – панель фильтров
        filter_frame = ttk.LabelFrame(main_panel, text="Фильтры и визуализация", width=270)
        filter_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        filter_frame.pack_propagate(False)

        self.var_stability = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="Только устойчивые решения",
                        variable=self.var_stability).pack(anchor=tk.W, pady=2)

        self.var_filter_error = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="Фильтр по ошибке",
                        variable=self.var_filter_error).pack(anchor=tk.W, pady=(10, 2))

        err_criteria_frame = ttk.Frame(filter_frame)
        err_criteria_frame.pack(fill=tk.X, pady=2)
        ttk.Label(err_criteria_frame, text="Критерий:").pack(side=tk.LEFT)
        self.error_criteria = ttk.Combobox(err_criteria_frame, values=["max_err", "l2_err", "mae"],
                                           state="readonly", width=8)
        self.error_criteria.pack(side=tk.LEFT, padx=5)
        self.error_criteria.set("max_err")

        threshold_frame = ttk.Frame(filter_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(threshold_frame, text="Порог:").pack(side=tk.LEFT)
        self.error_threshold = tk.StringVar(value="0.01")
        ttk.Entry(threshold_frame, textvariable=self.error_threshold, width=10).pack(side=tk.LEFT, padx=5)

        # Галочка автоматической визуализации
        self.var_visualize = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_frame, text="Визуализировать подходящие",
                        variable=self.var_visualize).pack(anchor=tk.W, pady=(10, 2))

        ttk.Separator(filter_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        self.animate_btn = ttk.Button(filter_frame, text="Анимация", command=self.open_animation)
        self.animate_btn.pack(pady=5)

        # ------------------ Блоки параметров ------------------
        ttk.Label(self.scrollable_frame, text="Параметры модели", font=('', 9, 'bold')).grid(row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(5,2))
        self.model_params_frame = ttk.Frame(self.scrollable_frame)
        self.model_params_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=5)

        ttk.Label(self.scrollable_frame, text="Параметры схемы", font=('', 9, 'bold')).grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=(10,2))
        self.scheme_params_frame = ttk.Frame(self.scrollable_frame)
        self.scheme_params_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=5)

        self.model_param_rows = []
        self.scheme_param_rows = []
        self.model_param_set = set()
        self.scheme_param_set = set()

        self._populate_scheme_params()
        self._populate_model_params()

        # ------------------ Таблица результатов ------------------
        table_frame = ttk.LabelFrame(self, text="Результаты")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

        self.tree = ttk.Treeview(table_frame, show='headings')
        self.tree.pack(fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll_y.set)

        # ------------------ Панель управления внизу ------------------
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(ctrl_frame, text="Запустить перебор", command=self.run_sweep).pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(ctrl_frame, text="Остановить", command=self.stop_sweep, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Сохранить таблицу (CSV)", command=self.save_table).pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(ctrl_frame, mode='determinate', maximum=100, value=0)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(20, 5))
        self.progress_label = ttk.Label(ctrl_frame, text="0 / 0", width=12)
        self.progress_label.pack(side=tk.LEFT, padx=5)

        self.stop_requested = False
        self.sweep_running = False
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ... все остальные методы без изменений, кроме _sweep_thread ...
    def _model_name_by_cls(self, cls):
        for name, cl in self.models_dict.items():
            if cl == cls:
                return name
        return list(self.models_dict.keys())[0]

    def _populate_model_params(self):
        for widget in self.model_params_frame.winfo_children():
            widget.destroy()
        self.model_param_rows.clear()
        self.model_param_set.clear()

        param_info = self.model_cls.param_info()
        for i, (name, default, tip) in enumerate(param_info):
            self.model_param_set.add(name)
            self._create_param_row(self.model_params_frame, i, name, 'model', default, self.model_param_rows)

    def _populate_scheme_params(self):
        for widget in self.scheme_params_frame.winfo_children():
            widget.destroy()
        self.scheme_param_rows.clear()
        self.scheme_param_set.clear()

        scheme_params = [
            ('s', self.base_scheme_args.get('s', DEFAULT_S)),
            ('L', self.base_scheme_args.get('L', DEFAULT_L)),
            ('T', self.base_scheme_args.get('T', DEFAULT_T)),
            ('N', self.base_scheme_args.get('N', DEFAULT_N)),
            ('M', self.base_scheme_args.get('M', DEFAULT_M)),
        ]
        for k in ['x0', 't0']:
            if k in self.base_scheme_args:
                scheme_params.append((k, self.base_scheme_args[k]))

        for i, (name, default) in enumerate(scheme_params):
            self.scheme_param_set.add(name)
            self._create_param_row(self.scheme_params_frame, i, name, 'scheme', default, self.scheme_param_rows)

    def _create_param_row(self, parent_frame, row_idx, name, category, default_val, storage_list):
        lbl = ttk.Label(parent_frame, text=name)
        lbl.grid(row=row_idx, column=0, padx=5, pady=2, sticky="w")

        type_var = tk.StringVar(value="fixed")
        type_menu = ttk.OptionMenu(parent_frame, type_var, "fixed",
                                   "fixed", "range", "list",
                                   command=lambda val, r=row_idx: self._update_param_fields(storage_list[r]))
        type_menu.grid(row=row_idx, column=1, padx=5, pady=2)

        fields_frame = ttk.Frame(parent_frame)
        fields_frame.grid(row=row_idx, column=2, columnspan=3, sticky="ew", padx=5, pady=2)

        row_data = {
            'name': name,
            'category': category,
            'default_val': default_val,
            'type_var': type_var,
            'fields_frame': fields_frame,
            'entries': {},
            'row_idx': row_idx,
        }
        storage_list.append(row_data)
        self._update_param_fields(row_data)

    def _update_param_fields(self, row_data):
        frame = row_data['fields_frame']
        for widget in frame.winfo_children():
            widget.destroy()
        entries = {}
        typ = row_data['type_var'].get()
        default_val = row_data['default_val']

        if typ == "fixed":
            ttk.Label(frame, text="Значение:").pack(side=tk.LEFT, padx=2)
            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=15)
            entry.pack(side=tk.LEFT, padx=2)
            entries['value'] = var
        elif typ == "range":
            ttk.Label(frame, text="Начало:").pack(side=tk.LEFT, padx=2)
            var_start = tk.StringVar(value=str(default_val))
            ttk.Entry(frame, textvariable=var_start, width=10).pack(side=tk.LEFT, padx=2)
            entries['start'] = var_start

            ttk.Label(frame, text="Конец:").pack(side=tk.LEFT, padx=2)
            var_end = tk.StringVar(value=str(default_val))
            ttk.Entry(frame, textvariable=var_end, width=10).pack(side=tk.LEFT, padx=2)
            entries['end'] = var_end

            ttk.Label(frame, text="Шагов:").pack(side=tk.LEFT, padx=2)
            var_steps = tk.StringVar(value="3")
            ttk.Entry(frame, textvariable=var_steps, width=5).pack(side=tk.LEFT, padx=2)
            entries['steps'] = var_steps
        elif typ == "list":
            ttk.Label(frame, text="Список:").pack(side=tk.LEFT, padx=2)
            var_list = tk.StringVar(value=str(default_val))
            ttk.Entry(frame, textvariable=var_list, width=20).pack(side=tk.LEFT, padx=2)
            entries['list'] = var_list

        row_data['entries'] = entries

    def on_model_change(self, event=None):
        name = self.model_var.get()
        if name in self.models_dict:
            self.model_cls = self.models_dict[name]
            self._populate_model_params()

    def generate_combinations(self):
        all_rows = self.model_param_rows + self.scheme_param_rows
        sweep = []
        for row in all_rows:
            name = row['name']
            typ = row['type_var'].get()
            entries = row['entries']
            try:
                if typ == "fixed":
                    val = float(entries['value'].get())
                    sweep.append({name: [val]})
                elif typ == "range":
                    start = float(entries['start'].get())
                    end = float(entries['end'].get())
                    num = int(float(entries['steps'].get()))
                    if num < 2:
                        raise ValueError("Число шагов должно быть >= 2")
                    vals = [round(v, 10) for v in np.linspace(start, end, num).tolist()]
                    sweep.append({name: vals})
                elif typ == "list":
                    text = entries['list'].get()
                    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
                    if not vals:
                        raise ValueError("Список пуст")
                    sweep.append({name: vals})
            except ValueError as e:
                messagebox.showerror("Ошибка", f"Некорректные данные для параметра '{name}': {e}")
                return []

        base_mod = {}
        for name, default, _ in self.model_cls.param_info():
            base_mod[name] = default
        base_sch = dict(self.base_scheme_args)

        from itertools import product
        keys = [list(d.keys())[0] for d in sweep]
        value_lists = [list(d.values())[0] for d in sweep]
        combos = []
        for values in product(*value_lists):
            mod = base_mod.copy()
            sch = base_sch.copy()
            for k, v in zip(keys, values):
                if k in self.model_param_set:
                    mod[k] = v
                elif k in self.scheme_param_set:
                    sch[k] = v
            combos.append((mod, sch))
        return combos

    def run_sweep(self):
        self.solutions_data.clear()
        self.solutions_errors.clear()
        combos = self.generate_combinations()
        if not combos:
            return

        self.stop_requested = False
        self.sweep_running = True
        self.stop_btn['state'] = tk.NORMAL
        self.tree.delete(*self.tree.get_children())

        all_param_names = [row['name'] for row in self.model_param_rows + self.scheme_param_rows]
        cols = list(dict.fromkeys(all_param_names))
        cols += ['max_err', 'l2_err', 'mae', 'M/N', 'h/delta']
        self.tree['columns'] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, minwidth=70, width=80, stretch=True)

        total = len(combos)
        self.progress['maximum'] = total
        self.progress['value'] = 0
        self.progress_label['text'] = f"0 / {total}"

        current_model_cls = self.model_cls
        current_method = METHODS[self.method_var.get()]
        filter_stability = self.var_stability.get()
        filter_error = self.var_filter_error.get()
        error_criteria = self.error_criteria.get()
        try:
            error_threshold = float(self.error_threshold.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный порог ошибки")
            self._on_sweep_finished()
            return

        do_visualize = self.var_visualize.get()

        threading.Thread(target=self._sweep_thread,
                         args=(combos, cols, total, current_model_cls, current_method,
                               filter_stability, filter_error, error_criteria, error_threshold,
                               do_visualize),
                         daemon=True).start()

    def _sweep_thread(self, combos, cols, total, model_cls, method,
                      filter_stability, filter_error, error_criteria, error_threshold,
                      do_visualize):
        try:
            for i, (model_args, scheme_args) in enumerate(combos):
                if self.stop_requested:
                    self.after(0, lambda: messagebox.showinfo("Остановлено",
                                                              f"Перебор остановлен. Выполнено {i} из {total}."))
                    break

                model = model_cls(**model_args)
                s = scheme_args['s']
                L = scheme_args['L']
                T = scheme_args['T']
                N = int(scheme_args['N'])
                M = int(scheme_args['M'])
                x0 = scheme_args.get('x0', 0.0)
                t0 = scheme_args.get('t0', 0.0)

                solver = DelayPDESolver(model, method, x0=x0, t0=t0, s=s, L=L, T=T, N=N, M=M)
                A, cfg = solver.solve(verbose=False)
                h = cfg.h
                delta = cfg.delta
                max_err, l2_err, mae = compute_error(A, cfg)

                add_to_table = True

                if filter_stability:
                    is_stable, _, _ = check_stability(cfg)   # теперь передаём cfg
                    if not is_stable:
                        add_to_table = False

                if add_to_table and filter_error:
                    err_val = None
                    if error_criteria == 'max_err':
                        err_val = max_err
                    elif error_criteria == 'l2_err':
                        err_val = l2_err
                    elif error_criteria == 'mae':
                        err_val = mae
                    if err_val is not None and err_val >= error_threshold:
                        add_to_table = False

                # Визуализация, если решение прошло фильтр и галочка включена
                if add_to_table and do_visualize:
                    # Создаём событие, которое будет установлено при закрытии окна
                    viz_done = threading.Event()
                    # Планируем открытие окна в главном потоке
                    self.after(0, lambda _A=A, _cfg=cfg, _model=model, m_args=model_args, s_args=scheme_args,
                                         mx=max_err, l2=l2_err, m=mae, done=viz_done:
                    self.master.open_plot_window(_A, _cfg, _model, m_args, s_args, mx, l2, m,
                                                 on_close_callback=lambda: done.set()))
                    # Ждём, пока окно не будет закрыто
                    # При этом проверяем, не запрошена ли остановка перебора
                    while not viz_done.is_set():
                        if self.stop_requested:
                            # Если пользователь нажал "Остановить", выходим, чтобы не висеть вечно
                            viz_done.set()  # прерываем ожидание
                            break
                        viz_done.wait(timeout=0.1)  # ждём с маленьким таймаутом для проверки stop_requested

                if add_to_table:
                    row = self._build_result_row(model_args, scheme_args, N, M, h, delta,
                                                 max_err, l2_err, mae, cols)
                    self.after(0, self._add_result_row, row, A, cfg, max_err, l2_err, mae)

                self.after(0, self._update_progress, i + 1)
        except Exception as e:
            self.after(0, messagebox.showerror, "Ошибка перебора", str(e))
        finally:
            self.after(0, self._on_sweep_finished)

    def _build_result_row(self, model_args, scheme_args, N, M, h, delta,
                          max_err, l2_err, mae, cols):
        row = []
        for c in cols:
            if c in model_args:
                row.append(model_args[c])
            elif c in scheme_args:
                if c in ('N', 'M'):
                    row.append(int(scheme_args[c]))
                else:
                    row.append(scheme_args[c])
            elif c == 'max_err':
                row.append(f"{max_err:.6f}")
            elif c == 'l2_err':
                row.append(f"{l2_err:.6f}")
            elif c == 'mae':
                row.append(f"{mae:.6f}")
            elif c == 'M/N':
                row.append(f"{M/N:.4f}" if N != 0 else "")
            elif c == 'h/delta':
                row.append(f"{h/delta:.4f}" if delta != 0 else "")
            else:
                row.append("")
        return row

    def _on_sweep_finished(self):
        self.sweep_running = False
        self.stop_btn['state'] = tk.DISABLED
        self.progress.stop()

    def stop_sweep(self):
        self.stop_requested = True
        self.stop_btn['state'] = tk.DISABLED

    def _update_progress(self, value):
        self.progress['value'] = value
        total = self.progress['maximum']
        self.progress_label['text'] = f"{value} / {total}"

    def _add_result_row(self, row, A=None, cfg=None, max_err=None, l2_err=None, mae=None):
        self.tree.insert('', tk.END, values=row)
        if A is not None and cfg is not None and np.all(np.isfinite(A)):
            self.solutions_data.append((A, cfg))
            if max_err is not None:
                self.solutions_errors.append((max_err, l2_err, mae))

    def save_table(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filepath:
            return
        columns = self.tree['columns']
        rows = [self.tree.item(child)['values'] for child in self.tree.get_children()]
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        messagebox.showinfo("Сохранено", f"Таблица сохранена в {filepath}")

    def open_animation(self):
        if not self.solutions_data:
            messagebox.showinfo("Нет данных", "Нет рассчитанных решений для анимации.")
            return
        AnimationDialog(self, self.solutions_data, self.solutions_errors)

    def on_close(self):
        if self.sweep_running:
            if messagebox.askyesno("Подтверждение", "Идёт перебор. Закрыть окно перебора?"):
                self.stop_requested = True
                self.destroy()
        else:
            self.destroy()
