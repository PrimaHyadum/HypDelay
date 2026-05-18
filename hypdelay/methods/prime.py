from hypdelay.core.base import PDEMethod

class PrimeMethod(PDEMethod):
    def get_current_extrapolation(self, cfg, u_prev, u_prev2):
        delta = cfg.delta
        s = cfg.s
        return ((delta + s) * u_prev - s * u_prev2) / delta

    def get_delayed_value(self, cfg, history, A, t_current, x_idx):
        tau = cfg.tau
        t0 = cfg.t0
        delta = cfg.delta

        t_delayed = t_current - tau

        if t_delayed < t0:

            return None  # специальное значение, означающее, что нужно использовать u_current
        else:
            idx = int(round((t_delayed - t0) / delta))
            idx = max(0, min(idx, len(A) - 1))
            return A[idx][x_idx]