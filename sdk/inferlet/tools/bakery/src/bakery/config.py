"""Configuration management for Bakery."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import toml

from . import path as path_utils


@dataclass
class ConfigFile:
    """Bakery configuration file structure."""

    registry_token: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "ConfigFile":
        """Load configuration from a TOML file."""
        content = path.read_text()
        data = toml.loads(content)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        """Save configuration to a TOML file."""
        data = {k: v for k, v in asdict(self).items() if v is not None}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(toml.dumps(data))


def get_token() -> Optional[str]:
    """Load the registry token from config if available."""
    config_path = path_utils.get_config_path()
    if config_path.exists():
        config = ConfigFile.load(config_path)
        return config.registry_token
    return None
