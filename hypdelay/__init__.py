"""
hypdelay - библиотека для численного решения гиперболических уравнений
в частных производных с запаздыванием (солитонные модели, тестовые уравнения).
"""

__version__ = "0.1.0"

# Основные классы ядра
from hypdelay.core.solver import DelayPDESolver
from hypdelay.core.config import Config
from hypdelay.core.base import PDEModel, PDEMethod

# Все модели
from hypdelay.models import (
    SineGordonKinkModel,
    Phi4KinkModel,
    DoubleSineGordonKinkModel,
    HyperbolicModel,
)

# Все численные методы
from hypdelay.methods import (
    PWLinearInterpExtrap,
    PrimeMethod,
    PrimeHybridMethod,
)

# Утилиты
from hypdelay.utils import (
    compute_error,
    plot_solution,
    check_stability,
    save_all_results,
)

# Графический интерфейс
from hypdelay.gui import Application

__all__ = [
    # классы
    "DelayPDESolver",
    "Config",
    "PDEModel",
    "PDEMethod",
    # модели
    "SineGordonKinkModel",
    "Phi4KinkModel",
    "DoubleSineGordonKinkModel",
    "HyperbolicModel",
    # методы
    "PWLinearInterpExtrap",
    "PrimeMethod",
    "PrimeHybridMethod",
    # утилиты
    "compute_error",
    "plot_solution",
    "check_stability",
    "save_all_results",
    # GUI
    "Application",
]

def run_gui():
    """Запуск графического интерфейса пользователя."""
    app = Application()
    app.mainloop()