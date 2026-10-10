"""会计恒等式检验：跑完一个场景，报告五条恒等式的残差（相对 GDP）。

    python validate/check_identities.py --scene level2_mixed

五条恒等式：
  goods   : K_{t+1} = (1-delta) K_t + Y_t - C_t            商品市场
  fund    : F_{t+1} = (1+r_t) F_t + 缴费 - 待遇              基金收支
  budget  : sum N (c + a') = sum N x                        家庭总预算
  bequest : tr_t N_t = (1+r_t) × 死者资产                    遗产分配
  assets  : 上年总储蓄 = 死者资产 + 存活者带入资产            资产守恒
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner import Runner  # noqa: E402
from utils.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="level2_mixed")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()

    cfg = load_config(args.scene, [f"checks.identities=false"])  # 自己汇总，不让环境中途报错
    runner = Runner(cfg)
    runner.burn_in()
    df = runner.run(args.T)
    cols = ["res_goods", "res_fund", "res_budget", "res_bequest", "res_assets"]
    print(f"scene={args.scene}  years={len(df)}  triggers={int(df.trigger.sum())}  "
          f"policy changes={int((df.tau_p.diff().abs() > 0).sum())}")
    print(df[cols].abs().agg(["max", "mean"]).T.to_string(float_format=lambda v: f"{v:.2e}"))
    worst = df[cols].abs().max().max()
    ok = worst < args.tol
    print(f"max |residual| / GDP = {worst:.2e}  ->  {'PASS' if ok else 'FAIL'} (tol {args.tol:.0e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
