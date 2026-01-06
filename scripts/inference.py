#!/usr/bin/env python
"""Inference script for running predictions on new data.

Usage:
    python scripts/inference.py checkpoint=path/to/checkpoint.ckpt input=path/to/images/
    python scripts/inference.py checkpoint=path/to/checkpoint.ckpt input=path/to/video.mp4
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_model(checkpoint_path: str, cfg: DictConfig) -> Any:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        cfg: Hydra configuration object.

    Returns:
        Loaded model in eval mode.
    """
    from scripts.train import DetectionModule

    model = DetectionModule.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
    )
    model.eval()
    return model


def preprocess_images(
    image_paths: List[str], cfg: DictConfig
) -> Dict[str, torch.Tensor]:
    """Preprocess images for inference.

    Args:
        image_paths: List of image file paths.
        cfg: Hydra configuration object.

    Returns:
        Dictionary with preprocessed image tensor.
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img_size = cfg.dataset.img_size
        img = cv2.resize(img, (img_size[1], img_size[0]))

        # Normalize
        img = img.astype(np.float32)
        mean = np.array(cfg.dataset.img_norm_cfg.mean)
        std = np.array(cfg.dataset.img_norm_cfg.std)
        img = (img - mean) / std

        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        images.append(img)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()

    # Add batch dimension: (N, C, H, W) -> (1, N, C, H, W)
    images = images.unsqueeze(0)

    return {'images': images}


def visualize_results(
    images: List[str],
    predictions: Dict[str, torch.Tensor],
    output_dir: str,
) -> None:
    """Visualize prediction results.

    Args:
        images: List of input image paths.
        predictions: Model predictions.
        output_dir: Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # TODO: Implement visualization
    # - Draw 3D bounding boxes projected to images
    # - Overlay map predictions
    # - Show depth estimation
    print(f'Results saved to: {output_dir}')


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """Main inference function.

    Args:
        cfg: Hydra configuration object.
    """
    print(OmegaConf.to_yaml(cfg))

    # Check required arguments
    if not hasattr(cfg, 'checkpoint') or cfg.checkpoint is None:
        raise ValueError('Please provide checkpoint path: checkpoint=path/to/ckpt')

    if not hasattr(cfg, 'input') or cfg.input is None:
        raise ValueError('Please provide input path: input=path/to/images/')

    # Load model
    print(f'Loading model from: {cfg.checkpoint}')
    model = load_model(cfg.checkpoint, cfg)

    # Find input images
    input_path = Path(cfg.input)
    if input_path.is_dir():
        image_paths = sorted(input_path.glob('*.jpg')) + sorted(input_path.glob('*.png'))
        image_paths = [str(p) for p in image_paths]
    else:
        image_paths = [str(input_path)]

    print(f'Found {len(image_paths)} images')

    # Run inference
    device = next(model.parameters()).device
    with torch.no_grad():
        batch = preprocess_images(image_paths, cfg)
        batch = {k: v.to(device) for k, v in batch.items()}
        predictions = model(batch)

    # Visualize results
    output_dir = cfg.get('output', 'outputs/inference')
    visualize_results(image_paths, predictions, output_dir)


if __name__ == '__main__':
    main()
