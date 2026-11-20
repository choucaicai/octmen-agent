"""Load portable defaults and optional machine-local overrides."""
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def config_section(name):
    defaults_path = ROOT / 'configs/config.yaml.example'
    local_path = ROOT / 'configs/config.yaml'
    settings = {}
    for path in (defaults_path, local_path):
        if path == local_path and not path.exists():
            continue
        document = yaml.safe_load(path.read_text())
        if document is None:
            continue
        if not isinstance(document, dict) or not isinstance(document.get(name, {}), dict):
            raise ValueError(f'{path}: expected a mapping for {name}')
        settings.update(document.get(name, {}))
    return settings


def resolve_path(value, *, relative_to=None):
    if value is None:
        return None
    path = Path(os.path.expandvars(os.fspath(value))).expanduser()
    return ((relative_to or ROOT) / path).resolve()
