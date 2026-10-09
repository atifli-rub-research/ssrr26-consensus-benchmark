from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from benchmark.observations.core.types import Crop
from typing import List

class Cropper(ABC):
    @abstractmethod
    def generate_cells(self, image, metadata) -> List[Crop]:
        pass


class ObservationModel(ABC):
    @abstractmethod
    def observe(self, crop, drone_id):
        pass


class CorruptionModel(ABC):
    @abstractmethod
    def corrupt(self, image: Image.Image, severity: float) -> Image.Image:
        pass

    @staticmethod
    def _validate_severity(severity: float):
        if not (0.0 <= severity <= 1.0):
            raise ValueError("severity must be in the range [0.0, 1.0]")

    @staticmethod
    def _to_numpy(image: Image.Image) -> np.ndarray:
        return np.asarray(image.convert("RGB")).astype(np.float32)

    @staticmethod
    def _to_pil(array: np.ndarray) -> Image.Image:
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _lerp(min_value: float, max_value: float, severity: float) -> float:
        """
        Linear interpolation between min_value and max_value.

        severity = 0.0 -> min_value
        severity = 1.0 -> max_value
        """
        return min_value + severity * (max_value - min_value)


class Detector(ABC):
    @abstractmethod
    def predict(self, image):
        pass


class ByzantineModel(ABC):
    @abstractmethod
    def corrupt_prediction(self, prediction, drone_id):
        pass


class ConsensusModel(ABC):
    @abstractmethod
    def fuse(self, predictions):
        pass
