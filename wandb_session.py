from typing import Any, Dict, Optional
from enum import Enum
import wandb as _wandb


class WandbMode(str, Enum):
    OFF = "off"
    ONLINE = "online"
    SWEEP = "sweep"

class WandbSession:
    def __init__(self, mode: WandbMode, project: str, name: str, config: Dict[str, Any]):
        self.mode = mode
        self.active = False
        self.run = None
        self.project = project
        self.name = name
        self.config = config

    def __enter__(self):
        if self.mode == WandbMode.OFF:
            return self
        if _wandb is None:
            print("[WARN] wandb not installed; continuing without logging.")
            return self
        # For sweeps, wandb agent will pass a config; we merge ours as defaults
        self.run = _wandb.init(project=self.project, config=self.config, reinit=True)
        self.active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active and _wandb is not None:
            _wandb.finish()
        self.active = False
        self.run = None

    def log(self, *args, **kwargs):
        if self.active and _wandb is not None:
            _wandb.log(*args, **kwargs)

    @property
    def cfg(self) -> Dict[str, Any]:
        """Return live wandb.config when available (sweep overrides), else local config dict."""
        if self.active and _wandb is not None:
            return dict(_wandb.config)
        return self.config