from dataclasses import dataclass
from pathlib import Path


@dataclass
class InferenceConfig:
    model_path: Path
    label_map_path: Path
    threshold: float
