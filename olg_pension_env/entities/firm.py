"""总量生产：Cobb-Douglas，生产率有趋势和 AR(1) 冲击，要素价格等于边际产出。"""
import numpy as np
import pandas as pd


class Firm:
    def __init__(self, cfg, pop, root):
        self.alpha = float(cfg.alpha)
        self.delta = float(cfg.delta)
        self.g = float(cfg.g)
        self.rho = float(cfg.rho_z)
        self.sigma = float(cfg.sigma_z)
        self.Z0 = float(cfg.Z0)
        self.shocks_on = bool(cfg.shocks_on)
        eff = pd.read_csv(root / cfg.efficiency_file)
        emap = dict(zip(eff["real_age"].astype(int), eff["efficiency"].astype(float)))
        self.e = np.array([emap[idx + pop.entry_age] for idx in range(pop.J)])  # 年龄效率 e_j
        self.reset()

    def reset(self):
        self.z = 0.0

    def draw_shock(self, rng, on=True):
        eps = rng.standard_normal() if on else 0.0
        self.z = self.rho * self.z + self.sigma * eps

    def Z(self, t):
        return self.Z0 * np.exp(self.g * t + self.z)

    def produce(self, K, L, Z):
        """返回 (Y, r, w)。r 已扣除折旧，w 是每效率单位劳动的工资。"""
        if K <= 0 or L <= 0:
            raise RuntimeError(f"资本或劳动非正: K={K:.4g}, L={L:.4g}")
        Y = Z * K ** self.alpha * L ** (1.0 - self.alpha)
        r = self.alpha * Y / K - self.delta
        w = (1.0 - self.alpha) * Y / L
        return Y, r, w
