from PIL import Image

from .common_corruptions import (
    CorruptionModel,
    GaussianNoise,
    MotionBlur,
    Snow,
    Pixelate,
    DefocusBlur,
)

_CORRUPTION_TYPES = {
    "gaussian_noise": GaussianNoise,
    "motion_blur": MotionBlur,
    "snow": Snow,
    "pixelate": Pixelate,
    "defocus": DefocusBlur,
}


def create_corruption(
    name: str,
    *,
    seed: int | None = None,
) -> CorruptionModel:
    try:
        cls = _CORRUPTION_TYPES[name]
    except KeyError:
        raise ValueError(
            f"Unknown corruption '{name}'. "
            f"Available: {', '.join(_CORRUPTION_TYPES)}"
        )

    return cls(seed=seed)


def apply_corruptions(
    image: Image.Image,
    corruption_config: dict,
    seed: int | None = None,
) -> Image.Image:

    corrupted = image.copy()

    for corruption_name, severity in corruption_config.items():

        if severity <= 0.0:
            continue

        corruption = create_corruption(corruption_name, seed=seed)

        corrupted = corruption.corrupt(
            corrupted,
            severity=severity,
        )

    return corrupted
