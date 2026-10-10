"""家庭：出生队列 × 行为类型 的代表性个体。所有数组形状为 (J, K)。

a        期初人均资产（上一年储蓄，由存活者带入）
x        本年可支配资源
c, a_next  本年消费、本年末储蓄
s        本年实际储蓄比例
s_peer_prev  上一年处在同一年龄的人（各类型按占比加权）的实际储蓄比例，形状 (J,)；
         S 型本年决策时能看到的同龄人储蓄率。reset 后的第一年没有上一年，为 NaN。
kappa_e  对自己退休后替代率的预期；预期养老金 b_e = kappa_e × 当年在职者平均工资。
         预期按替代率而不是按金额形成，是为了避免自适应预期在工资增长趋势下永远滞后。
         行为类型的主要区别就在 kappa_e 怎么更新。
"""
import numpy as np

TYPE_NAMES = ("R", "P", "A", "S")


class Households:
    def __init__(self, cfg, pop):
        self.J = pop.J
        self.types = [str(t) for t in cfg.types]
        for t in self.types:
            if t not in TYPE_NAMES:
                raise ValueError(f"未知行为类型 {t}")
        self.K = len(self.types)
        self.pi = np.array([float(v) for v in cfg.shares])
        if self.pi.shape != (self.K,) or abs(self.pi.sum() - 1.0) > 1e-12:
            raise ValueError("households.shares 必须与 types 等长且和为 1")
        self.labor = float(cfg.labor)
        self.sigma = float(cfg.sigma)
        self.varphi = float(cfg.varphi)
        self.chi = float(cfg.chi)
        self.gamma_A = float(cfg.gamma_A)
        self.fund_ref = float(cfg.fund_confidence_ref)
        self.omega_s = float(cfg.omega_s)
        self.s_max = float(cfg.s_max)
        self.reset()

    def reset(self):
        z = lambda: np.zeros((self.J, self.K))
        self.a, self.a_next, self.x, self.c, self.s, self.u, self.kappa_e = z(), z(), z(), z(), z(), z(), z()
        self.worker = np.ones(self.J, dtype=bool)
        self.s_peer_prev = np.full(self.J, np.nan)
        self._has_history = False

    # ---- 每年的步骤 ----
    def age_forward(self):
        """队列老化一岁：上一年的储蓄、预期由存活者带到下一个年龄；新进入者资产为 0。
        同龄人储蓄率不随队列移动：记下上一年各年龄的平均实际储蓄比例，供 S 型本年参照。"""
        top = np.zeros((1, self.K))
        self.a = np.vstack([top, self.a_next[:-1]])
        self.kappa_e = np.vstack([top, self.kappa_e[:-1]])
        if self._has_history:
            self.s_peer_prev = self.s @ self.pi

    def expected_benefit(self, wbar):
        return self.kappa_e * wbar

    def resources(self, r, w, tau_p, tau_l, JR, b, tr, e):
        """x = (1+r) a + (1 - tau_l - tau_p) w e_j l 1[在职] + b 1[退休] + tr"""
        idx = np.arange(self.J)
        self.worker = idx < JR
        labor_income = (1.0 - tau_l - tau_p) * w * e * self.labor * self.worker
        pension_income = b * (~self.worker)
        self.x = (1.0 + r) * self.a + labor_income[:, None] + pension_income[:, None] + tr
        return self.x

    def consume(self, s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.s_max)
        s = np.where(self.x > 0, s, 0.0)
        self.s = s
        self._has_history = True
        self.c = (1.0 - s) * self.x
        self.a_next = s * self.x

    def utility(self):
        c = np.maximum(self.c, 1e-12)
        if abs(self.sigma - 1.0) < 1e-12:
            uc = np.log(c)
        else:
            uc = c ** (1.0 - self.sigma) / (1.0 - self.sigma)
        l = self.labor * self.worker
        ul = self.chi * l ** (1.0 + 1.0 / self.varphi) / (1.0 + 1.0 / self.varphi)
        self.u = uc - ul[:, None]
        return self.u

    # ---- 预期 ----
    def set_entrant_expectation(self, kappa):
        self.kappa_e[0, :] = kappa

    def signal(self, kappa, kappa_real, fund_ratio, payg_rate):
        """A、S 型据以修正预期的“信号”：他们认为制度实际能兑现的替代率。警戒线规则：

        基金占产出比 >= fund_ref ：信号 = 实际替代率
        基金占产出比 <  fund_ref ：信号 = 只靠当年缴费能发的替代率 = 缴费率 / 抚养比（不超过实际替代率）

        即基金低于警戒线时不再相信承诺，改按当年缴费够发多少来预期。fund_ref <= 0 时不看基金。
        返回 (信号, 是否高于警戒线)。"""
        if self.fund_ref <= 0 or fund_ratio >= self.fund_ref:
            return kappa_real, 1.0
        return min(kappa_real, payg_rate), 0.0

    def update_expectations(self, kappa, kappa_signal):
        """kappa: 当前规则的替代率；kappa_signal: 见 signal()。
        R/P: 预期 = 当前规则；A: 向信号靠拢；S: A 型更新后再与同龄人平均混合。"""
        peer = self.kappa_e @ self.pi  # (J,) 同龄人（各类型按占比加权）的预期，更新前
        new = self.kappa_e.copy()
        for k, typ in enumerate(self.types):
            if typ in ("R", "P"):
                new[:, k] = kappa
            elif typ == "A":
                new[:, k] = self.kappa_e[:, k] + self.gamma_A * (kappa_signal - self.kappa_e[:, k])
            elif typ == "S":
                own = self.kappa_e[:, k] + self.gamma_A * (kappa_signal - self.kappa_e[:, k])
                new[:, k] = (1.0 - self.omega_s) * own + self.omega_s * peer
        self.kappa_e = new
