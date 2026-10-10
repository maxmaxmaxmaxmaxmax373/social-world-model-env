"""第一类环境核心：按设计文档 1.6.1 的八步顺序推进一年。

一年分两个阶段调用（因为家庭要在看到本年价格之后才决策）：
    hh_obs = env.begin_year(policy_action)   # 步骤 1-4：人口、冲击、政策、期内出清
    pol_obs, rewards, info = env.end_year(s) # 步骤 5-8：家庭执行、基金结算、预期更新、奖励
"""
from pathlib import Path

import numpy as np

from entities.firm import Firm
from entities.households import Households
from entities.pension import PensionAuthority
from entities.population import Population
from env.observations import household_obs, policy_obs

ROOT = Path(__file__).resolve().parent.parent

IDENTITY_KEYS = ("res_goods", "res_fund", "res_budget", "res_bequest", "res_assets")


class OLGPensionEnv:
    def __init__(self, cfg, root=ROOT):
        self.cfg = cfg
        self.pop = Population(cfg.population, root)
        self.firm = Firm(cfg.production, self.pop, root)
        self.hh = Households(cfg.households, self.pop)
        self.pension = PensionAuthority(cfg.pension, self.pop.entry_age)
        self.H = int(cfg.pension.H)
        self.tau_l = float(cfg.fiscal.tau_l)
        self.F0_over_Y = float(cfg.pension.F0_over_Y)
        self.check = bool(cfg.checks.identities)
        self.tol = float(cfg.checks.tol)
        self.burn_in = False
        self.reset()

    # ------------------------------------------------------------------ 初始化
    def reset(self):
        self.rng = np.random.default_rng(int(self.cfg.seed))
        self.t = 0
        self.N_prev = self.pop.stationary(float(self.cfg.population.N0))
        self.hh.reset()
        self.pension.reset()
        self.firm.reset()
        # 初始资本猜测：K/Y = initial_KY_guess，按人均均匀分配
        JR = self.pension.JR
        L = float((self.N_prev[:JR] * self.firm.e[:JR]).sum() * self.hh.labor)
        KY = float(self.cfg.production.initial_KY_guess)
        K = (KY * self.firm.Z0) ** (1.0 / (1.0 - self.firm.alpha)) * L
        self.hh.a_next[:] = K / self.N_prev.sum()
        self.pension.F = self.F0_over_Y * K / KY
        self._fresh = True
        self.cur, self.last = None, None
        self.records, self.cohort_records = [], []
        self.pol_return, self.last_decision_t = 0.0, 0
        return policy_obs(self)

    def set_burn_in(self, flag):
        self.burn_in = bool(flag)

    def finish_burn_in(self):
        """燃烧期结束：把水平变量按平衡增长路径缩回 Z0 对应的尺度，时间归零。"""
        scale = (self.firm.Z(self.t) / self.firm.Z0) ** (1.0 / (1.0 - self.firm.alpha))
        self.hh.a_next /= scale
        self.pension.F /= scale
        if self.last is not None:
            self.last = {k: (v / scale if k in ("Y", "w", "K", "C", "S") else v) for k, v in self.last.items()}
        self.firm.z = 0.0
        self.t = 0
        self.pension.triggers = 0
        self.records, self.cohort_records = [], []
        self.pol_return, self.last_decision_t = 0.0, 0
        self.burn_in = False
        return policy_obs(self)

    def is_decision_year(self):
        return (not self.burn_in) and (self.t % self.H == 0)

    # ------------------------------------------------------------------ 步骤 1-4
    def begin_year(self, policy_action=None):
        hh, pop, firm, pen = self.hh, self.pop, self.firm, self.pension
        t = self.t
        pi = hh.pi

        # 1. 人口更新；上一年的储蓄由存活者带入，死者的部分留作遗产
        S_prev = float((self.N_prev[:, None] * pi * hh.a_next).sum())
        N, deaths, births = pop.step(self.N_prev, year=None if self.burn_in else t)
        dead_assets = float((deaths[:, None] * pi * hh.a_next).sum())
        hh.age_forward()
        surv_assets = float((N[:, None] * pi * hh.a).sum())

        # 2. 生产率冲击
        firm.draw_shock(self.rng, on=firm.shocks_on and not self.burn_in)
        Z = firm.Z(t)

        # 3. 政策（每 H 年一次；燃烧期锁定）
        if self.is_decision_year() and policy_action is not None:
            pen.apply_action(policy_action)
            self.last_decision_t = t
        JR = pen.JR

        # 4. 期内出清：劳动无弹性，不需要迭代
        K = S_prev + pen.F
        worker = np.arange(pop.J) < JR
        L = float((N[worker] * firm.e[worker]).sum() * hh.labor)
        N_W, N_R, N_tot = float(N[worker].sum()), float(N[~worker].sum()), float(N.sum())
        Y, r, w = firm.produce(K, L, Z)
        plan = pen.plan(r, w, L, N_W, N_R)
        if self.burn_in:  # 燃烧期：基金占产出比冻结，不触发自动平衡
            plan.update(b=plan["b_rule"], benefits=plan["b_rule"] * N_R,
                        F_next=self.F0_over_Y * Y, trigger=0)
        tr = (1.0 + r) * dead_assets / N_tot
        hh.resources(r, w, pen.tau_p, self.tau_l, JR, plan["b"], tr, firm.e)
        if self._fresh:
            hh.kappa_e[:] = pen.kappa
            self._fresh = False
        hh.set_entrant_expectation(pen.kappa)

        self.N = N
        self.cur = dict(t=t, N_tot=N_tot, N_W=N_W, N_R=N_R, births=births, JR=JR,
                        S_prev=S_prev, dead_assets=dead_assets, surv_assets=surv_assets,
                        K=K, L=L, Z=Z, Y=Y, r=r, w=w, tr=tr, F=pen.F, **plan)
        return household_obs(self)

    # ------------------------------------------------------------------ 步骤 5-8
    def end_year(self, s):
        hh, pen, cur = self.hh, self.pension, self.cur
        if cur is None:
            raise RuntimeError("请先调用 begin_year")
        N, pi = self.N, hh.pi

        # 5. 家庭执行
        hh.consume(s)
        C = float((N[:, None] * pi * hh.c).sum())
        S = float((N[:, None] * pi * hh.a_next).sum())
        X = float((N[:, None] * pi * hh.x).sum())

        # 6. 基金结算（自动平衡已在 begin_year 的 plan 中决定）
        F_next = cur["F_next"]
        K_next = S + F_next
        pen.triggers += cur["trigger"]

        # 7. 更新预期。A、S 型看的信号：基金高于警戒线时 = 实际替代率，否则 = 只靠当年缴费能发的替代率
        kappa_real = cur["b"] / cur["wbar"] if cur["wbar"] > 0 else pen.kappa
        payg_rate = pen.tau_p * cur["N_W"] / cur["N_R"] if cur["N_R"] > 0 else pen.kappa
        kappa_signal, confidence = hh.signal(pen.kappa, kappa_real, F_next / cur["Y"], payg_rate)
        hh.update_expectations(pen.kappa, kappa_signal)
        workers = N[hh.worker]
        ke_by_type = (workers[:, None] * hh.kappa_e[hh.worker]).sum(axis=0) / workers.sum()  # 在职者的平均预期，分类型

        # 8. 奖励
        u = hh.utility()
        W = float((N[:, None] * pi * u).sum() / cur["N_tot"])
        pol_reward = W - pen.phi_F * cur["trigger"]
        self.pol_return += pen.beta ** (cur["t"] - self.last_decision_t) * pol_reward

        # 会计恒等式残差（相对 GDP）
        Y, K, r = cur["Y"], cur["K"], cur["r"]
        res = dict(
            res_goods=(K_next - ((1.0 - self.firm.delta) * K + Y - C)) / Y,
            res_fund=(F_next - ((1.0 + r) * cur["F"] + cur["contrib"] - cur["benefits"])) / Y,
            res_budget=((C + S) - X) / Y,
            res_bequest=(cur["tr"] * cur["N_tot"] - (1.0 + r) * cur["dead_assets"]) / Y,
            res_assets=(cur["S_prev"] - cur["dead_assets"] - cur["surv_assets"]) / Y,
        )
        if self.check and not self.burn_in:
            bad = {k: v for k, v in res.items() if abs(v) > self.tol}
            if bad:
                raise RuntimeError(f"第 {cur['t']} 年会计恒等式不成立: {bad}")

        row = dict(
            t=cur["t"], N=cur["N_tot"], births=cur["births"], N_W=cur["N_W"], N_R=cur["N_R"],
            DR=cur["N_R"] / cur["N_W"], Z=cur["Z"], z=self.firm.z, Y=Y, K=K, L=cur["L"], C=C, S=S,
            KY=K / Y, CY=C / Y, I=K_next - (1.0 - self.firm.delta) * K,
            r=r, w=cur["w"], wbar=cur["wbar"], F=cur["F"], F_next=F_next, FY=cur["F"] / Y,
            contrib=cur["contrib"], benefits=cur["benefits"], b_rule=cur["b_rule"], b=cur["b"],
            b_cut=1.0 - cur["b"] / cur["b_rule"] if cur["b_rule"] > 0 else 0.0,
            tau_p=pen.tau_p, kappa=pen.kappa, retire_age=pen.retire_age_real, trigger=cur["trigger"],
            tr=cur["tr"], W=W, pol_reward=pol_reward,
            s_mean_workers=float((hh.s[hh.worker] @ pi).mean()),
            kappa_real=kappa_real, kappa_signal=kappa_signal, confidence=confidence,
            **{f"ke_{typ}": float(ke_by_type[k]) for k, typ in enumerate(hh.types)},
            **res,
        )
        if not self.burn_in:
            self.records.append(row)
            if bool(self.cfg.save_cohorts):
                self.cohort_records.append(dict(t=cur["t"], N=N.copy(), c=hh.c.copy(), a=hh.a.copy(),
                                                s=hh.s.copy(), kappa_e=hh.kappa_e.copy(), x=hh.x.copy()))

        # 状态推进
        pen.F = F_next
        self.N_prev = N
        self.last = row
        self.t += 1
        self.cur = None
        rewards = dict(households=u, pension=pol_reward)
        return policy_obs(self), rewards, row
