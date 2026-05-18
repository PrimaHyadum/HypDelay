from hypdelay.core.base import PDEMethod


class PrimeHybridMethod(PDEMethod):
    def get_current_extrapolation(self, cfg, u_prev, u_prev2):
        return 2 * u_prev - u_prev2

    def get_delayed_value(self, cfg, history, A, t_current, x_idx):
        tau = cfg.tau
        t0 = cfg.t0
        delta = cfg.delta
        eps = cfg.eps

        t_delayed = t_current - tau

        if t_delayed < t0:

            # Используем историю
            t_prev, t_prev2, u_prev, u_prev2 = history.get_interval(t_delayed, x_idx)

            # Линейная интерполяция
            if t_prev2 - t_prev < eps:
                w = (t_delayed - t_prev) / (t_prev2 - t_prev)
            else:
                w = 0.0
            return u_prev * (1 - w) + u_prev2 * w

        else:
            idx = int(round((t_delayed - t0) / delta))
            idx = max(0, min(idx, len(A) - 1))
            return A[idx][x_idx]