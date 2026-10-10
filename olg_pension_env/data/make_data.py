"""生成默认输入数据：分年龄死亡率、模型进入率（生育）、年龄效率曲线。

用法（在 olg_pension_env 目录下）：
    python data/make_data.py                       # 用内置系数生成三个 CSV
    python data/make_data.py --scf <SCF csv 路径>  # 年龄效率改为从 SCF 2022 微观数据重新估计
        # 当前布局下的路径：../../EconGym/agents/data/advanced_scfp2022_1110.csv

生成的 CSV 已随仓库提供，正常使用不需要重跑。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# CDC 2022 临时死亡率，每 10 万人（Ahmad et al. 2023）。与 EconGym 使用的是同一张表。
CDC_2022 = [
    (0, 0, 560.0), (1, 4, 28.0), (5, 14, 15.3), (15, 24, 79.5), (25, 34, 163.4),
    (35, 44, 255.4), (45, 54, 453.3), (55, 64, 992.1), (65, 74, 1978.7),
    (75, 84, 4708.2), (85, 100, 14389.6),
]

# 从 EconGym 自带的 SCF 2022 数据估出的 log 工资对年龄的二次式系数（LF=1, WAGEINC>0, 21-70 岁, WGT 加权）
SCF_EFFICIENCY_BETA = (0.0711681673, -0.00129555716)


def make_survival(ages=range(21, 101)):
    """把 10 岁一档的死亡率按 log 线性插值到单岁。85+ 档取 92 岁为中点，之后按最后一段斜率外推。"""
    mids = np.array([(lo + hi) / 2 for lo, hi, _ in CDC_2022], dtype=float)
    mids[-1] = 92.0
    logm = np.log(np.array([rate for *_, rate in CDC_2022]) / 1e5)
    slope = (logm[-1] - logm[-2]) / (mids[-1] - mids[-2])
    rows = []
    for a in ages:
        lm = np.interp(a, mids, logm) if a <= mids[-1] else logm[-1] + slope * (a - mids[-1])
        rows.append((int(a), min(float(np.exp(lm)), 1.0)))
    return pd.DataFrame(rows, columns=["real_age", "mortality"])


def make_fertility(J=80, center=31, sd=5.0, lo=21, hi=46):
    """人造进入率形状：钟形，中心在模型年龄 31（父母 30 岁生育），支撑 21..46。
    绝对水平由 Population.scale_fertility 按目标增长率重新缩放，这里只给形状。"""
    j = np.arange(1, J + 1)
    f = np.exp(-0.5 * ((j - center) / sd) ** 2)
    f[(j < lo) | (j > hi)] = 0.0
    return pd.DataFrame({"model_age": j, "entry_rate": f / f.sum()})


def asfr_to_entry_rates(asfr_by_real_age, entry_age=21, J=80, child_survival=0.99):
    """把真实数据的分年龄生育率（每名妇女每年活产数）换成模型进入率。

    父母在真实年龄 b 生育，孩子 21 年后进入模型，此时父母真实年龄 b+21，
    模型年龄 j = b + 21 - entry_age + 1。人口不分性别，所以除以 2。
    """
    f = np.zeros(J)
    for b, rate in asfr_by_real_age.items():
        j = int(b) + 21 - entry_age + 1
        if 1 <= j <= J:
            f[j - 1] = float(rate) / 2.0 * child_survival
    return pd.DataFrame({"model_age": np.arange(1, J + 1), "entry_rate": f})


def make_efficiency(beta=SCF_EFFICIENCY_BETA, scf_csv=None, ages=range(21, 101)):
    """年龄效率 e(age) = exp(b1 (age-21) + b2 (age-21)^2)，按 21-64 岁均值归一化为 1。"""
    if scf_csv is not None:
        df = pd.read_csv(scf_csv)
        d = df[(df.LF == 1) & (df.WAGEINC > 0) & (df.AGE >= 21) & (df.AGE <= 70)]
        age = d.AGE.values.astype(float)
        y = np.log(d.WAGEINC.values)
        w = np.sqrt(d.WGT.values)
        X = np.column_stack([np.ones_like(age), age - 21, (age - 21) ** 2])
        coef = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
        beta = (float(coef[1]), float(coef[2]))
    a = np.array(list(ages), dtype=float)
    e = np.exp(beta[0] * (a - 21) + beta[1] * (a - 21) ** 2)
    e = e / e[(a >= 21) & (a <= 64)].mean()
    return pd.DataFrame({"real_age": a.astype(int), "efficiency": e}), beta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scf", default=None, help="SCF 2022 csv 路径（可选）")
    args = ap.parse_args()
    make_survival().to_csv(HERE / "survival_cdc2022.csv", index=False)
    make_fertility().to_csv(HERE / "fertility_synthetic.csv", index=False)
    eff, beta = make_efficiency(scf_csv=args.scf)
    eff.to_csv(HERE / "efficiency_scf2022.csv", index=False)
    print("written: survival_cdc2022.csv, fertility_synthetic.csv, efficiency_scf2022.csv")
    print("efficiency beta:", beta, " peak real age:", 21 - beta[0] / (2 * beta[1]))
