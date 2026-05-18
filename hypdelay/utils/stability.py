import numpy as np
from hypdelay.core.config import Config

def check_stability(cfg: Config):
    """
    Проверяет, удовлетворяет ли выбранный весовой коэффициент s условию устойчивости.

    :param cfg: конфигурация
    :return: (is_stable, s_min) – кортеж (стабильна ли схема, минимальное s)
    """
    sigma = cfg.rh
    s_min = 1.0 / 4.0 * (1.0 - 1.0 / sigma)
    is_stable = cfg.s > s_min

    return is_stable, s_min, sigma