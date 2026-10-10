"""养老金当局：带结余基金的现收现付制，含自动平衡机制与政策调整约束。"""
import numpy as np

KEYS = ("tau_p", "kappa", "retire_age_real")


class PensionAuthority:
    def __init__(self, cfg, entry_age):
        self.entry_age = int(entry_age)
        self.init = {"tau_p": float(cfg.tau_p), "kappa": float(cfg.kappa),
                     "retire_age_real": int(cfg.retire_age_real)}
        self.bounds = {k: (float(cfg.bounds[k][0]), float(cfg.bounds[k][1])) for k in KEYS}
        self.max_step = {k: float(cfg.max_step[k]) for k in KEYS}
        self.phi_F = float(cfg.phi_F)
        self.beta = float(cfg.beta)
        self.reset()

    def reset(self):
        for k, v in self.init.items():
            setattr(self, k, v)
        self.F = 0.0
        self.triggers = 0
        self.last_action = dict(self.init)

    @property
    def JR(self):
        """第一个退休的模型年龄下标：真实年龄 >= retire_age_real 即退休。"""
        return int(self.retire_age_real) - self.entry_age

    def apply_action(self, action):
        """每次决策的变化不超过 max_step，且落在 bounds 内；退休年龄取整。"""
        if action is None:
            return
        for k in KEYS:
            if k not in action:
                continue
            cur = float(getattr(self, k))
            new = float(np.clip(float(action[k]), cur - self.max_step[k], cur + self.max_step[k]))
            new = float(np.clip(new, *self.bounds[k]))
            if k == "retire_age_real":
                new = int(round(new))
            setattr(self, k, new)
        self.last_action = {k: getattr(self, k) for k in KEYS}

    def plan(self, r, w, L, N_W, N_R):
        """基金流量与自动平衡。F_next = (1+r)F + tau_p w L - b N_R；若为负则削减 b 使 F_next = 0。"""
        wbar = w * L / N_W if N_W > 0 else 0.0
        b_rule = self.kappa * wbar
        contrib = self.tau_p * w * L
        F_next = (1.0 + r) * self.F + contrib - b_rule * N_R
        b, trigger = b_rule, 0
        if F_next < 0.0:
            avail = (1.0 + r) * self.F + contrib
            b = avail / N_R if N_R > 0 else 0.0
            F_next, trigger = 0.0, 1
        return dict(wbar=wbar, b_rule=b_rule, b=b, contrib=contrib, benefits=b * N_R,
                    F_next=F_next, trigger=trigger)
