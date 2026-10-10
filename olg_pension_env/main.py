"""入口：python main.py --scene level2_R [--T 200] [--set households.tilt=0.9 ...]"""
import argparse
import json

from runner import Runner
from utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="base", help="cfg/ 下的场景文件名（不含 .yaml）")
    ap.add_argument("--T", type=int, default=None, help="主运行年数，默认取配置")
    ap.add_argument("--set", nargs="*", default=None, help="覆盖参数，如 households.tilt=0.9")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.scene, args.set)
    runner = Runner(cfg)
    runner.burn_in()
    runner.run(args.T)
    print(json.dumps(runner.summary(), indent=2, ensure_ascii=False))
    if not args.no_save:
        print("saved to", runner.save())


if __name__ == "__main__":
    main()
