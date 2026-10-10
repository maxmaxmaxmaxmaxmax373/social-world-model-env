from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent


def load_config(scene=None, overrides=None):
    """加载 cfg/base.yaml，并与 cfg/<scene>.yaml 深合并；overrides 为 "a.b=c" 形式的列表。"""
    cfg = OmegaConf.load(ROOT / "cfg" / "base.yaml")
    if scene and scene != "base":
        path = ROOT / "cfg" / f"{scene}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"找不到场景文件 {path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
        cfg.scene = scene
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
