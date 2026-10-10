"""年龄队列人口（Leslie 结构）。

所有数组都按下标 idx = 0..J-1 存放。下标 idx 对应模型年龄 j = idx+1、真实年龄 idx + entry_age
（默认 idx=0 是 21 岁，idx=79 是 100 岁）。下面的说明一律用下标：
psi[idx] : 从下标 idx-1 活到下标 idx 的概率（psi[0] 未使用，置 1）
f[idx]   : 下标 idx 的每个人在下一年带来的新进入者数（进入率）
"""
import numpy as np
import pandas as pd


class Population:
    def __init__(self, cfg, root):
        self.J = int(cfg.J)
        self.entry_age = int(cfg.entry_age_real)

        surv = pd.read_csv(root / cfg.survival_file)
        mort = dict(zip(surv["real_age"].astype(int), surv["mortality"].astype(float)))
        self.psi = np.ones(self.J)
        for idx in range(1, self.J):
            # 真实年龄 idx-1+entry_age 的人在这一年内死亡的概率为 mort[...]
            self.psi[idx] = 1.0 - mort[idx - 1 + self.entry_age]
        self.L = np.cumprod(self.psi)  # L[idx]: 从进入（下标 0）活到下标 idx 的累计存活率，L[0]=1

        fert = pd.read_csv(root / cfg.fertility_file)
        f = np.zeros(self.J)
        for j, v in zip(fert["model_age"].astype(int), fert["entry_rate"].astype(float)):
            if 1 <= j <= self.J:
                f[j - 1] = v
        self.growth_target = float(cfg.growth_target)
        self.f = self.scale_fertility(f, self.growth_target)

        # cum_surv[j, k]: 下标 j 的人活到下标 k 的概率（k >= j）
        self.cum_surv = np.zeros((self.J, self.J))
        for j in range(self.J):
            self.cum_surv[j, j:] = self.L[j:] / self.L[j]
        # 期望剩余寿命（含当年）
        self.horizon = self.cum_surv.sum(axis=1)

        shock = cfg.fertility_shock
        self.shock_year = None if shock.year is None else int(shock.year)
        self.shock_mult = float(shock.multiplier)

    # ---- 起点：让人口稳定所需的进入率，以及稳定时的年龄分布 ----
    def scale_fertility(self, f, lam):
        """把进入率整体放大或缩小，使人口每年恰好变为原来的 lam 倍（lam=1 即总人口不变）。

        lam=1 时的条件是 sum(f * L) = 1：一个人一生中平均恰好带来一个新进入者。
        一般情形是 sum(f[idx] * L[idx] / lam^(idx+1)) = 1。"""
        idx = np.arange(self.J)
        denom = np.sum(f * self.L * lam ** (-(idx + 1.0)))
        return f / denom

    def stationary(self, N_total):
        """人口稳定时各年龄的人数，总数为 N_total。

        lam=1 时各年龄人数正比于活到该年龄的概率 L；lam<1（人口收缩）时年轻人相对更少。"""
        idx = np.arange(self.J)
        N = self.L * self.growth_target ** (-idx.astype(float))
        return N / N.sum() * N_total

    # ---- 每年调用：把人口推进一年 ----
    def step(self, N, year=None):
        """从 N 推进一年。返回 (N_next, deaths, births)。
        deaths[idx]: 今年处在下标 idx、没有活到明年的人数（最高年龄全部死亡）。
        year: 新队列进入的年份；等于 shock_year 时乘以 shock_mult。"""
        mult = self.shock_mult if (self.shock_year is not None and year == self.shock_year) else 1.0
        births = mult * float(self.f @ N)
        N_next = np.empty_like(N)
        N_next[0] = births
        N_next[1:] = self.psi[1:] * N[:-1]
        deaths = np.empty_like(N)
        deaths[:-1] = N[:-1] * (1.0 - self.psi[1:])
        deaths[-1] = N[-1]
        return N_next, deaths, births
