from dataclasses import dataclass, field
from PIL.Image import Image

@dataclass
class Crop:
    cell_id: str
    image: Image
    row: int
    col: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Prediction:
    cell_id: str
    drone_id: str
    label: int
    confidence: float
    metadata: dict = field(default_factory=dict)

@dataclass
class Drone:
    drone_id: str
    model: str
    corruptions: dict

@dataclass
class Observation:
    drone_id: str
    tile_index: int
    clean_image: Image
    corrupted_image: Image
    corruptions: dict
