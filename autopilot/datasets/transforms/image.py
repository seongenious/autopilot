"""Data transforms for image-based training (panoptic, depth, etc.)."""

from typing import Any, Dict, List, Tuple

import numpy as np


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, **data: Dict) -> Dict:
        for t in self.transforms:
            data = t(**data)
        return data


class RandomCrop:
    """Random crop for image and all targets.

    Handles targets with different shapes:
        - (H, W): 2D arrays like semantic masks
        - (C, H, W): 3D arrays like center/offset/depth targets

    Args:
        crop_size: (height, width) tuple.
    """

    def __init__(self, crop_size: Tuple[int, int]):
        self.crop_size = crop_size

    def __call__(self, image: np.ndarray, **targets: Any) -> Dict:
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_size

        # Random crop position
        if h > crop_h:
            top = np.random.randint(0, h - crop_h)
        else:
            top = 0
        if w > crop_w:
            left = np.random.randint(0, w - crop_w)
        else:
            left = 0

        # Crop image (H, W, C)
        image = image[top:top + crop_h, left:left + crop_w]

        # Crop all targets
        result = {"image": image}
        for key, value in targets.items():
            if value is None:
                result[key] = None
            elif value.ndim == 2:
                # (H, W) array
                result[key] = value[top:top + crop_h, left:left + crop_w]
            elif value.ndim == 3:
                # (C, H, W) array
                result[key] = value[:, top:top + crop_h, left:left + crop_w]
            else:
                # Pass through unchanged
                result[key] = value

        return result


class RandomHorizontalFlip:
    """Random horizontal flip.

    Args:
        prob: Probability of flipping.
        offset_x_key: Key for x-offset that needs sign negation (default: None).
    """

    def __init__(self, prob: float = 0.5, offset_x_key: str = "offset_target"):
        self.prob = prob
        self.offset_x_key = offset_x_key

    def __call__(self, image: np.ndarray, **targets: Any) -> Dict:
        result = {"image": image, **targets}

        if np.random.random() < self.prob:
            # Flip image (H, W, C)
            result["image"] = np.ascontiguousarray(image[:, ::-1])

            # Flip all targets
            for key, value in targets.items():
                if value is None:
                    continue
                elif value.ndim == 2:
                    # (H, W) array
                    result[key] = np.ascontiguousarray(value[:, ::-1])
                elif value.ndim == 3:
                    # (C, H, W) array
                    result[key] = np.ascontiguousarray(value[:, :, ::-1])

            # Negate x-offset if present (offset_target[1] is x-offset)
            if self.offset_x_key in result and result[self.offset_x_key] is not None:
                offset = result[self.offset_x_key]
                if offset.ndim == 3 and offset.shape[0] >= 2:
                    offset[1] = -offset[1]

        return result


class Normalize:
    """Normalize image with mean and std.

    Args:
        mean: RGB mean values.
        std: RGB std values.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray, **targets: Any) -> Dict:
        # Normalize image (H, W, C)
        image = (image - self.mean) / self.std

        return {"image": image, **targets}
