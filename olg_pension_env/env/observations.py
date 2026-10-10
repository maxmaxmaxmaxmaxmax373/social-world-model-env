"""各角色的观测。家庭观测在期内出清之后生成（含本年价格）；政策观测用上一年末的值。"""
import numpy as np


def household_obs(env):
    cur, hh, pop = env.cur, env.hh, env.pop
    idx = np.arange(pop.J)
    psi_next = np.append(pop.psi[1:], 0.0)  # 活到下一岁的概率，最高年龄为 0
    return dict(
        t=env.t, idx=idx, real_age=idx + pop.entry_age, worker=hh.worker.copy(),
        a=hh.a.copy(), x=hh.x.copy(), kappa_e=hh.kappa_e.copy(), b_e=hh.expected_benefit(cur["wbar"]),
        s_peer_prev=hh.s_peer_prev.copy(),
        w=cur["w"], r=cur["r"], z=env.firm.z,
        tau_p=env.pension.tau_p, tau_l=env.tau_l, JR=cur["JR"],
        b=cur["b"], b_rule=cur["b_rule"], tr=cur["tr"],
        e=env.firm.e.copy(), labor=hh.labor, psi_next=psi_next,
        cum_surv=pop.cum_surv, horizon=pop.horizon,
        types=list(hh.types), pi=hh.pi.copy(),
    )


def policy_obs(env):
    pop, pen = env.pop, env.pension
    N = env.N_prev  # 上一年末（= 本年初更新前）的人口
    JR = pen.JR
    N_W, N_R = N[:JR].sum(), N[JR:].sum()
    bins = []
    for lo in range(0, pop.J, 10):  # 按 10 岁一档：真实年龄 21-30, 31-40, ..., 91-100
        bins.append(N[lo:lo + 10].sum() / N.sum())
    last = env.last if env.last is not None else {}
    Y_last = last.get("Y", np.nan)
    return dict(
        t=env.t, DR=N_R / N_W if N_W > 0 else np.inf, age_bins=np.array(bins),
        F_over_Y=pen.F / Y_last if Y_last and np.isfinite(Y_last) else np.nan,
        w=last.get("w", np.nan), r=last.get("r", np.nan), z=env.firm.z,
        last_action=dict(pen.last_action), triggers=pen.triggers,
        bounds=dict(pen.bounds), max_step=dict(pen.max_step),
    )
