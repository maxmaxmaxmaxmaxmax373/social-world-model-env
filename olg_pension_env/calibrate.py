"""校准：用二分法调一个参数，使燃烧期稳态的资本产出比等于目标值（设计文档：约 3）。

    python calibrate.py --scene level2_R --param households.tilt --target 3.0
    python calibrate.py --scene level1_fixed --param households.fixed_saving_workers --lo 0.5 --hi 0.95

设计文档里这一目标原本靠贴现因子 beta 匹配；rule-based 没有 beta，改为调储蓄规则里的参数。
"""
import argparse

from runner import Runner
from utils.config import load_config


def steady_ky(scene, param, value):
    cfg = load_config(scene, [f"{param}={value}", "checks.identities=false"])
    runner = Runner(cfg)
    runner.burn_in()
    return runner.env.last["KY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="level2_R")
    ap.add_argument("--param", default="households.tilt")
    ap.add_argument("--target", type=float, default=3.0)
    ap.add_argument("--lo", type=float, default=0.5)
    ap.add_argument("--hi", type=float, default=1.6)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    f_lo, f_hi = steady_ky(args.scene, args.param, args.lo), steady_ky(args.scene, args.param, args.hi)
    print(f"{args.param}={args.lo}: K/Y={f_lo:.3f}   {args.param}={args.hi}: K/Y={f_hi:.3f}")
    if (f_lo - args.target) * (f_hi - args.target) > 0:
        raise SystemExit("目标值不在区间内，请调整 --lo/--hi")
    lo, hi = args.lo, args.hi
    for _ in range(args.iters):
        mid = 0.5 * (lo + hi)
        f_mid = steady_ky(args.scene, args.param, mid)
        if (f_mid - args.target) * (f_lo - args.target) > 0:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
        if abs(f_mid - args.target) < 1e-4:
            break
    print(f"结果: {args.param} = {mid:.5f}  ->  K/Y = {f_mid:.4f}")
    print(f"写入场景文件的格式：\n{args.param.split('.')[0]}:\n  {args.param.split('.')[1]}: {mid:.5f}")


if __name__ == "__main__":
    main()
