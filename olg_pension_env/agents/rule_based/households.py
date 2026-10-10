"""家庭的规则型决策：返回储蓄比例 s，形状 (J, K)。

fixed       : 在职者固定储蓄比例；退休者把资源按期望剩余寿命平摊。
expectation : 把 手头资源 + 预期未来工资 + 预期未来养老金 平摊到期望剩余寿命，
              预期养老金 b_e 由行为类型决定（见 entities/households.py）。
              P 型多消费 present_bias_markup；S 型的储蓄比例再向上一年同龄人的平均靠拢。
"""
import numpy as np


class RuleHouseholds:
    def __init__(self, cfg, env=None):
        hc = cfg.households
        self.rule = str(hc.rule)
        self.s_fixed = float(hc.fixed_saving_workers)
        self.tilt = float(hc.tilt)
        self.markup = float(hc.present_bias_markup)
        self.omega_s = float(hc.omega_s)
        self.s_max = float(hc.s_max)

    def get_action(self, obs):
        if self.rule == "fixed":
            return self._fixed(obs)
        if self.rule == "expectation":
            return self._expectation(obs)
        raise ValueError(f"未知家庭规则 {self.rule}")

    # ---- 规则一：固定储蓄比例 ----
    def _fixed(self, obs):
        J, K = obs["x"].shape
        s = np.full((J, K), self.s_fixed)
        drawdown = 1.0 - 1.0 / obs["horizon"]  # 退休者：c = x / 期望剩余寿命
        s[~obs["worker"], :] = drawdown[~obs["worker"]][:, None]
        return np.clip(s, 0.0, self.s_max)

    # ---- 规则二：按预期平摊 ----
    @staticmethod
    def horizons(obs):
        cs, JR, J = obs["cum_surv"], obs["JR"], obs["x"].shape[0]
        inc = (1.0 - obs["tau_l"] - obs["tau_p"]) * obs["w"] * obs["e"] * obs["labor"]
        inc = np.where(np.arange(J) < JR, inc, 0.0)
        PV_wage = np.array([cs[j, j + 1:] @ inc[j + 1:] for j in range(J)])     # 未来工资（当前价格）
        H_pension = np.array([cs[j, max(j + 1, JR):].sum() for j in range(J)])  # 未来领取养老金的期望年数
        return obs["horizon"], PV_wage, H_pension

    def _expectation(self, obs):
        x, b_e, types = obs["x"], obs["b_e"], obs["types"]
        H, PV_wage, H_pension = self.horizons(obs)
        c = self.tilt * (x + PV_wage[:, None] + b_e * H_pension[:, None]) / H[:, None]
        for k, typ in enumerate(types):
            if typ == "P":
                c[:, k] *= 1.0 + self.markup
        c = np.minimum(c, x)                        # 借贷约束 a' >= 0
        c = np.maximum(c, (1.0 - self.s_max) * x)   # s <= s_max
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(x > 0, 1.0 - c / x, 0.0)
        s_peer = obs["s_peer_prev"]                 # 上一年同龄人（各类型按占比加权）的实际储蓄率
        for k, typ in enumerate(types):
            if typ == "S":
                peer = np.where(np.isnan(s_peer), s[:, k], s_peer)  # 燃烧期第一年没有上一年：不混合
                s[:, k] = (1.0 - self.omega_s) * s[:, k] + self.omega_s * peer
        return np.clip(s, 0.0, self.s_max)
