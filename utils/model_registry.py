"""Resolve a self-contained policy package for training and saved checkpoints."""
import json
from pathlib import Path
from types import SimpleNamespace


MODEL_PACKAGE = 'model_octmem_agent'
MODEL_CLASS = 'OpenVLAForActionPrediction'


def resolve_model(module=MODEL_PACKAGE, class_name=MODEL_CLASS):
    if module != MODEL_PACKAGE or not class_name.isidentifier():
        raise ValueError(f'Expected a model class from {MODEL_PACKAGE}, got {module}.{class_name}')
    from model_octmem_agent import modeling_prismatic, processing_prismatic as processing

    model = getattr(modeling_prismatic, class_name)
    return SimpleNamespace(model=model, config=model.config_class,
                           image_processor=processing.PrismaticImageProcessor,
                           processor=processing.PrismaticProcessor,
                           auxiliaries=model.auxiliary_module_names)


def checkpoint_model(directory):
    config = json.loads((Path(directory) / 'config.json').read_text())
    package = config.get('memory_model_module')
    if package != MODEL_PACKAGE:
        raise ValueError(f'Expected a {MODEL_PACKAGE} checkpoint, got {package!r}: {directory}')
    return package, config.get('memory_model_class', MODEL_CLASS)


def stamp_model_config(config, module, class_name, auxiliaries):
    config.memory_model_module = module
    config.memory_model_class = class_name
    config.memory_auxiliary_modules = list(auxiliaries)
