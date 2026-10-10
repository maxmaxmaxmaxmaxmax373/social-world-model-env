"""画图：python plot.py --scene level2_mixed  ->  output/<scene>/overview.png, cohorts.png
（matplotlib 默认字体没有中文，图内标签用英文。）"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def overview(outdir):
    df = pd.read_csv(outdir / "series.csv")
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    ax = ax.ravel()
    ax[0].plot(df.t, df.Y); ax[0].set_yscale("log"); ax[0].set_title("GDP (log scale)")
    ax[1].plot(df.t, df.KY, label="K/Y"); ax[1].plot(df.t, df.CY, label="C/Y"); ax[1].legend(); ax[1].set_title("Capital-output and consumption-output ratios")
    ax[2].plot(df.t, df.DR, label="dependency ratio"); ax[2].set_title("Dependency ratio (line) and births (gray)")
    ax2 = ax[2].twinx(); ax2.plot(df.t, df.births, color="gray", alpha=0.6); ax2.set_ylabel("births")
    ax[3].plot(df.t, df.FY, label="F/Y"); ax[3].set_title("Fund / GDP; red dots = auto-balance triggered")
    trig = df[df.trigger > 0]; ax[3].scatter(trig.t, trig.FY, color="red", s=12)
    ax[4].plot(df.t, df.tau_p, label="tau_p (contribution rate)"); ax[4].plot(df.t, df.b / df.wbar, label="b / wbar (realized replacement)")
    ax[4].plot(df.t, df.kappa, "--", label="kappa (statutory replacement)")
    for typ, col in (("A", "tab:red"), ("S", "tab:purple")):  # 在职者的平均预期替代率
        if f"ke_{typ}" in df and (df[f"ke_{typ}"] - df.kappa).abs().max() > 1e-9:
            ax[4].plot(df.t, df[f"ke_{typ}"], color=col, lw=1.0, label=f"expected replacement, type {typ} (workers)")
    ax[4].legend(fontsize=7); ax[4].set_title("Policy, realized benefit and expectations")
    ax[5].plot(df.t, df.W); ax[5].set_title("Social welfare W_t (population-weighted utility)")
    for a in ax: a.set_xlabel("year"); a.grid(alpha=0.3)
    fig.suptitle(f"scene: {outdir.name}"); fig.tight_layout(); fig.savefig(outdir / "overview.png", dpi=130); plt.close(fig)


def cohorts(outdir, years=(0, 50, 100, 150, 199)):
    """分年龄剖面。消费和资产都除以当年的在职者平均工资 wbar：
    经济每年增长约 2%，不做这一步的话后面年份的曲线会把前面的压成一条平线，无法比较形状。"""
    p = outdir / "cohorts.npz"
    if not p.exists():
        return
    d = np.load(p, allow_pickle=True)
    wbar = pd.read_csv(outdir / "series.csv").set_index("t")["wbar"]
    types, pi, t_all = list(d["types"]), d["pi"], d["t"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ages = np.arange(d["c"].shape[1]) + 21
    for yr in years:
        if yr not in t_all: continue
        i = int(np.where(t_all == yr)[0][0])
        ax[0].plot(ages, d["c"][i] @ pi / wbar[yr], label=f"year {yr}")
        ax[1].plot(ages, d["a"][i] @ pi / wbar[yr], label=f"year {yr}")
    ax[0].set_title("Consumption by age / average wage"); ax[1].set_title("Assets by age / average wage")
    i = len(t_all) - 1
    for k, typ in enumerate(types):
        if pi[k] > 0: ax[2].plot(ages, d["s"][i][:, k], label=f"type {typ}")
    ax[2].set_title(f"Saving share s by type (year {int(t_all[i])})")
    for a in ax: a.set_xlabel("real age"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"scene: {outdir.name}"); fig.tight_layout(); fig.savefig(outdir / "cohorts.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="base")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--years", type=int, nargs="*", default=None, help="分年龄图里画哪几年，默认 0 50 100 150 199")
    args = ap.parse_args()
    outdir = Path(args.outdir or Path("output") / args.scene)
    overview(outdir)
    cohorts(outdir, tuple(args.years)) if args.years else cohorts(outdir)
    print("saved", outdir / "overview.png", outdir / "cohorts.png")
