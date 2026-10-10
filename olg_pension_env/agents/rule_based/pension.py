"""养老金当局的规则型决策：返回 dict(tau_p, kappa, retire_age_real)。环境负责裁剪幅度和范围。

fixed      : 沿用上一次的政策。
fund_ratio : 基金占产出比低于 low 上调缴费率 dtau，高于 high 下调 dtau；替代率与退休年龄不动。
"""


class RulePension:
    def __init__(self, cfg, env=None):
        pc = cfg.pension
        self.rule = str(pc.rule)
        self.low = float(pc.fund_ratio_rule.low)
        self.high = float(pc.fund_ratio_rule.high)
        self.dtau = float(pc.fund_ratio_rule.dtau)

    def get_action(self, obs):
        action = dict(obs["last_action"])
        if self.rule == "fixed":
            return action
        if self.rule == "fund_ratio":
            FY = obs["F_over_Y"]
            if FY == FY:  # 非 NaN
                if FY < self.low:
                    action["tau_p"] += self.dtau
                elif FY > self.high:
                    action["tau_p"] -= self.dtau
            return action
        raise ValueError(f"未知政策规则 {self.rule}")
