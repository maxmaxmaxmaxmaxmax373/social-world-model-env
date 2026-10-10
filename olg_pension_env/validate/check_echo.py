"""人口回声检验：只用人口模块。给平稳人口一次生育冲击，比较模拟出的回声周期和衰减率与
Leslie 矩阵第二特征值的理论值。

    python validate/check_echo.py [--mult 1.2] [--years 400]

理论：冲击后新进入者数的偏离 = sum_i c_i lambda_i^t v_i。主特征值 lambda_1 = 1 对应平稳分布；
模最大的复特征值对 lambda_2 决定回声：周期 = 2 pi / |arg lambda_2|，每年衰减 = |lambda_2|。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from entities.population import Population  # noqa: E402
from utils.config import load_config, ROOT  # noqa: E402


def leslie_eigenvalues(pop):
    """把人口更新规则写成一张 J×J 的表（第一行是进入率，对角线下方是存活率），
    求它的特征值，按模从大到小排序。只有这项检验用到，模拟本身不需要。"""
    A = np.zeros((pop.J, pop.J))
    A[0, :] = pop.f
    for i in range(1, pop.J):
        A[i, i - 1] = pop.psi[i]
    vals = np.linalg.eigvals(A)
    return vals[np.argsort(-np.abs(vals))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, default=1.2, help="第 1 年新进入者放大倍数")
    ap.add_argument("--years", type=int, default=400)
    ap.add_argument("--tol", type=float, default=0.05, help="周期与衰减率的相对误差容忍")
    args = ap.parse_args()

    cfg = load_config("base", ["population.growth_target=1.0",
                               "population.fertility_shock.year=1",
                               f"population.fertility_shock.multiplier={args.mult}"])
    pop = Population(cfg.population, ROOT)

    # 理论值
    vals = leslie_eigenvalues(pop)
    lam1 = vals[0]
    cplx = [v for v in vals[1:] if abs(v.imag) > 1e-9]
    lam2 = max(cplx, key=abs)
    period_th = 2 * np.pi / abs(np.angle(lam2))
    decay_th = abs(lam2) / abs(lam1)

    # 模拟。多跑一段（共 3000 年）只为得到长期水平。
    N = pop.stationary(10000.0)
    births_ss = float(pop.f @ N)
    births = []
    for year in range(1, 3001):
        N, _, b = pop.step(N, year=year)
        births.append(b)
    dev_all = np.array(births) / births_ss - 1.0

    # 主特征值为 1 时，一次性多进入的人会永久抬高人口水平（lambda_1 模态不衰减）。
    # 回声是围绕这个新水平的振荡，所以量衰减之前必须先减掉它。
    level = float(dev_all[-1])
    osc = dev_all[:args.years] - level

    # 第一个回声里还混有更快衰减的模态（周期 16 年、11 年等），从第二个回声开始量
    peaks, _ = find_peaks(osc)
    peaks = peaks[osc[peaks] > 0][1:7]
    spacing = np.diff(peaks)
    period_sim = float(spacing.mean())
    ratios = osc[peaks[1:]] / osc[peaks[:-1]]
    decay_sim = float(np.mean(ratios ** (1.0 / spacing)))

    err_p = abs(period_sim - period_th) / period_th
    err_d = abs(decay_sim - decay_th) / decay_th
    print(f"Leslie 主特征值        : {lam1.real:.6f}")
    print(f"第二特征值 (复)        : {lam2:.4f}")
    print(f"回声周期   理论 {period_th:7.2f} 年   模拟 {period_sim:7.2f} 年   相对误差 {err_p:.1%}")
    print(f"每年衰减   理论 {decay_th:7.4f}      模拟 {decay_sim:7.4f}      相对误差 {err_d:.1%}")
    print(f"每次回声相对上一次: 理论 {decay_th ** period_th:.3f}   模拟 {float(ratios.mean()):.3f}")
    print(f"新进入人数的永久抬升: {level:+.2%}   峰值年份: {(peaks + 1).tolist()}   半衰期约 {np.log(0.5) / np.log(decay_th):.0f} 年")
    ok = err_p < args.tol and err_d < args.tol
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
