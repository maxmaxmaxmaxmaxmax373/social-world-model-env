"""运行控制：燃烧期、主运行、保存结果。没有训练。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from agents.rule_based.households import RuleHouseholds
from agents.rule_based.pension import RulePension
from env.env_core import OLGPensionEnv
from env.observations import policy_obs


def build(cfg):
    env = OLGPensionEnv(cfg)
    return env, RuleHouseholds(cfg, env), RulePension(cfg, env)


class Runner:
    def __init__(self, cfg, env=None, hh_agent=None, pol_agent=None):
        self.cfg = cfg
        if env is None:
            env, hh_agent, pol_agent = build(cfg)
        self.env, self.hh_agent, self.pol_agent = env, hh_agent, pol_agent
        self.burn_in_years = 0

    def _year(self, policy_action):
        hh_obs = self.env.begin_year(policy_action)
        s = self.hh_agent.get_action(hh_obs)
        return self.env.end_year(s)

    def burn_in(self):
        """政策固定、冲击关闭、基金占比冻结，迭代到 K/Y 收敛，再把水平变量缩回初始尺度。"""
        years, tol = int(self.cfg.burn_in.years), float(self.cfg.burn_in.tol)
        self.env.set_burn_in(True)
        prev = None
        for i in range(years):
            _, _, row = self._year(None)
            if prev is not None and abs(row["KY"] - prev) < tol and i > 100:
                break
            prev = row["KY"]
        self.burn_in_years = i + 1
        return self.env.finish_burn_in()

    def run(self, T=None, pol_obs=None):
        """从环境当前状态开始跑 T 年（构造时已 reset；通常先调用 burn_in）。"""
        T = int(self.cfg.T if T is None else T)
        if self.env.burn_in:
            self.env.finish_burn_in()
        if pol_obs is None:
            pol_obs = policy_obs(self.env)
        for _ in range(T):
            action = self.pol_agent.get_action(pol_obs) if self.env.is_decision_year() else None
            pol_obs, _, _ = self._year(action)
        return self.series()

    def series(self):
        return pd.DataFrame(self.env.records)

    def summary(self):
        df = self.series()
        last = df.iloc[-1]
        return dict(
            scene=str(self.cfg.scene), years=int(len(df)), burn_in_years=int(self.burn_in_years),
            KY_start=float(df.KY.iloc[0]), KY_end=float(last.KY),
            r_mean=float(df.r.mean()), CY_mean=float(df.CY.mean()),
            DR_start=float(df.DR.iloc[0]), DR_end=float(last.DR),
            FY_start=float(df.FY.iloc[0]), FY_end=float(last.FY),
            triggers=int(df.trigger.sum()), benefit_cut_max=float(df.b_cut.max()),
            tau_p_end=float(last.tau_p), retire_age_end=float(last.retire_age),
            W_mean=float(df.W.mean()), pol_return=float(self.env.pol_return),
            max_abs_residual=float(df[[c for c in df.columns if c.startswith("res_")]].abs().max().max()),
        )

    def save(self, outdir=None):
        outdir = Path(outdir or Path(self.cfg.output_dir) / str(self.cfg.scene))
        outdir.mkdir(parents=True, exist_ok=True)
        df = self.series()
        df.to_csv(outdir / "series.csv", index=False)
        if self.env.cohort_records:
            packed = {k: np.stack([rec[k] for rec in self.env.cohort_records])
                      for k in ("N", "c", "a", "s", "kappa_e", "x")}
            packed["t"] = np.array([rec["t"] for rec in self.env.cohort_records])
            packed["types"] = np.array(self.env.hh.types)
            packed["pi"] = self.env.hh.pi
            np.savez_compressed(outdir / "cohorts.npz", **packed)
        with open(outdir / "summary.json", "w") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=False)
        OmegaConf.save(self.cfg, outdir / "config.yaml")
        return outdir
