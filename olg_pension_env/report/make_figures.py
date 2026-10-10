"""生成报告用的图：aging 场景与两个对照变体的比较。在 olg_pension_env 目录下运行：

    python report/make_figures.py

对照变体只在内存里运行，不写入 output/。
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import Runner  # noqa: E402
from utils.config import load_config  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"

VARIANTS = [
    ("aging (rule, with shock)", [], dict(color="tab:blue", lw=1.8)),
    ("rule, no fertility shock", ["population.fertility_shock.year=null"], dict(color="tab:orange", lw=1.2, ls="--")),
    ("fixed policy tau_p = 0.14", ["pension.rule=fixed"], dict(color="tab:red", lw=1.2)),
]


def simulate(overrides):
    runner = Runner(load_config("aging", overrides + ["save_cohorts=false"]))
    runner.burn_in()
    return runner.run(200)


def main():
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for label, overrides, style in VARIANTS:
        df = simulate(overrides)
        ax[0].plot(df.t, df.FY, label=label, **style)
        ax[1].plot(df.t, df.tau_p, label=label, **style)
        ax[2].plot(df.t, df.b / df.wbar, label=label, **style)
    ax[0].set_title("Fund / GDP")
    ax[0].axhline(0.2, color="gray", lw=0.6, ls=":"); ax[0].axhline(2.0, color="gray", lw=0.6, ls=":")
    ax[1].set_title("Contribution rate tau_p")
    ax[1].axhline(0.185, color="gray", lw=0.6, ls=":"); ax[1].text(2, 0.187, "balanced rate 0.185", fontsize=8, color="gray")
    ax[2].set_title("Realized replacement rate b / wbar (statutory 0.40)")
    for a in ax:
        a.set_xlabel("year"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "aging_counterfactual.png", dpi=130)
    plt.close(fig)
    for name in ("overview", "cohorts"):  # 这两张由 plot.py 生成，这里只复制
        src = ROOT / "output" / "aging" / f"{name}.png"
        if src.exists():
            shutil.copy(src, FIG / f"aging_{name}.png")
    print("saved", FIG / "aging_counterfactual.png")


if __name__ == "__main__":
    main()
