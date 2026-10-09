from benchmark.observations.core.interfaces import CorruptionModel
from typing import Optional
import numpy as np
from PIL import Image, ImageEnhance
import cv2

class GaussianNoise(CorruptionModel):
    """
    Gaussian-style corruption that smoothly transitions from:
        severity = 0.0 -> original image
        severity = 1.0 -> pure random noise
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def corrupt(self, image: Image.Image, severity: float) -> Image.Image:
        self._validate_severity(severity)

        if severity == 0.0:
            return image.copy()

        image_np = self._to_numpy(image)

        # Generate a completely random noise image.
        noise_image = self.rng.uniform(
            low=0.0,
            high=255.0,
            size=image_np.shape,
        )

        # Blend original image with random noise.
        #
        # severity = 0.0 -> fully original
        # severity = 1.0 -> fully random noise
        corrupted = (
            (1.0 - severity) * image_np +
            severity * noise_image
        )

        return self._to_pil(corrupted)

class MotionBlur(CorruptionModel):
    """
    Motion blur corruption with smooth severity scaling.

    severity = 0.0:
        Original image

    severity = 1.0:
        Extremely blurred image with most spatial detail destroyed
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def corrupt(self, image: Image.Image, severity: float) -> Image.Image:
        self._validate_severity(severity)

        if severity == 0.0:
            return image.copy()

        image_np = self._to_numpy(image)

        # Large maximum kernel to allow near-total destruction
        # of fine spatial detail at severity=1.0.
        kernel_size = int(round(
            self._lerp(
                min_value=3,
                max_value=151,
                severity=severity,
            )
        ))

        # Ensure odd kernel size.
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

        # Horizontal motion blur kernel.
        kernel[kernel_size // 2, :] = 1.0
        kernel /= kernel_size

        blurred = cv2.filter2D(
            image_np,
            ddepth=-1,
            kernel=kernel,
        )

        # Blend original and blurred image.
        #
        # severity = 0.0 -> original
        # severity = 1.0 -> fully blurred
        corrupted = (
            (1.0 - severity) * image_np +
            severity * blurred
        )

        return self._to_pil(corrupted)

class Snow(CorruptionModel):
    """
    Snow corruption where severity directly controls image coverage.

    severity = 0.0:
        Original image

    severity = 0.5:
        Approximately 50% of pixels covered by snow

    severity = 1.0:
        100% of pixels covered by snow
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def corrupt(self, image: Image.Image, severity: float) -> Image.Image:
        self._validate_severity(severity)

        if severity == 0.0:
            return image.copy()

        image_np = self._to_numpy(image)
        h, w, _ = image_np.shape

        # Severity directly controls the probability that a pixel is covered.
        snow_mask = self.rng.random((h, w)) < severity

        snow = np.full_like(image_np, 255.0)

        corrupted = image_np.copy()
        corrupted[snow_mask] = snow[snow_mask]

        return self._to_pil(corrupted)

class Pixelate(CorruptionModel):
    """
    Pixelation corruption with intuitive nonlinear severity scaling.

    severity = 0.0:
        Original image

    severity = 0.5:
        Clearly pixelated, but still partly interpretable

    severity = 1.0:
        Image is downsampled to 1x1, then upsampled back to original size.
        This removes essentially all spatial detail.
    """

    def __init__(self, seed: Optional[int] = None, exponent: float = 2.0):
        self.seed = seed
        self.exponent = exponent

    def corrupt(self, image: Image.Image, severity: float) -> Image.Image:
        self._validate_severity(severity)

        if severity == 0.0:
            return image.copy()

        original_w, original_h = image.size

        # Makes medium severity visually meaningful.
        #
        # severity = 0.0 -> effective_severity = 0.0
        # severity = 1.0 -> effective_severity = 1.0
        effective_severity = 1.0 - ((1.0 - severity) ** self.exponent)

        min_scale = 1.0 / max(original_w, original_h)

        scale = self._lerp(
            min_value=1.0,
            max_value=min_scale,
            severity=effective_severity,
        )

        small_size = (
            max(1, int(round(original_w * scale))),
            max(1, int(round(original_h * scale))),
        )

        small = image.resize(
            small_size,
            resample=Image.Resampling.BOX,
        )

        pixelated = small.resize(
            (original_w, original_h),
            resample=Image.Resampling.NEAREST,
        )

        return pixelated

class DefocusBlur(CorruptionModel):
    """
    Defocus blur corruption with smooth severity scaling.

    severity = 0.0:
        Original image

    severity = 1.0:
        Strongly defocused image with most fine spatial detail removed

    The severity controls both:
        1. the disk blur radius
        2. how much the blurred image replaces the original image
    """

    def __init__(self, seed: Optional[int] = None, max_radius: int = 64):
        self.seed = seed
        self.max_radius = max_radius

    def corrupt(self, image: Image.Image, severity: float) -> Image.Image:
        self._validate_severity(severity)

        if severity == 0.0:
            return image.copy()

        image_np = self._to_numpy(image)

        radius = int(round(
            self._lerp(
                min_value=1,
                max_value=self.max_radius,
                severity=severity,
            )
        ))

        radius = max(1, radius)

        kernel = self._disk_kernel(radius)

        blurred = cv2.filter2D(
            image_np,
            ddepth=-1,
            kernel=kernel,
        )

        # Blend original and defocused image.
        #
        # severity = 0.0 -> original
        # severity = 1.0 -> fully defocused
        corrupted = (
            (1.0 - severity) * image_np
            + severity * blurred
        )

        return self._to_pil(corrupted)

    @staticmethod
    def _disk_kernel(radius: int) -> np.ndarray:
        size = radius * 2 + 1

        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = x**2 + y**2 <= radius**2

        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[mask] = 1.0
        kernel /= kernel.sum()

        return kernel
