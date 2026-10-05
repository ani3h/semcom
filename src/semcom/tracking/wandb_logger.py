from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


def init_run(cfg: DictConfig, run_name: str, project: str = "semcom", **kwargs) -> Any:
    return wandb.init(
        project=project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        **kwargs,
    )
