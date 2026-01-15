"""Visualization script for panoptic segmentation."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from autopilot.datasets import CityscapesDataset, NUM_THINGS, NUM_STUFF
from autopilot.datasets.transforms import Compose, Normalize, RandomCrop
from autopilot.models.builder import build_model

# Import to register modules
import autopilot.models.backbones  # noqa: F401
import autopilot.models.necks  # noqa: F401
import autopilot.models.heads.perception2d  # noqa: F401
import autopilot.models.detectors  # noqa: F401


# Cityscapes color palette (things first, then stuff)
PALETTE = [
    # Things (0-7): person, rider, car, truck, bus, train, motorcycle, bicycle
    (220, 20, 60),    # person - red
    (255, 0, 0),      # rider - bright red
    (0, 0, 142),      # car - dark blue
    (0, 0, 70),       # truck - navy
    (0, 60, 100),     # bus - dark cyan
    (0, 80, 100),     # train - teal
    (0, 0, 230),      # motorcycle - blue
    (119, 11, 32),    # bicycle - maroon
    # Stuff (8-18): road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky
    (128, 64, 128),   # road - purple
    (244, 35, 232),   # sidewalk - pink
    (70, 70, 70),     # building - dark gray
    (102, 102, 156),  # wall - gray-purple
    (190, 153, 153),  # fence - light gray
    (153, 153, 153),  # pole - gray
    (250, 170, 30),   # traffic light - orange
    (220, 220, 0),    # traffic sign - yellow
    (107, 142, 35),   # vegetation - olive green
    (152, 251, 152),  # terrain - light green
    (70, 130, 180),   # sky - steel blue
]

CLASS_NAMES = [
    # Things
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    # Stuff
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
]


def colorize_semantic(semantic: np.ndarray) -> np.ndarray:
    """Convert semantic labels to color image.

    Args:
        semantic: (H, W) semantic labels.

    Returns:
        (H, W, 3) color image.
    """
    h, w = semantic.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, rgb in enumerate(PALETTE):
        mask = semantic == label_id
        color[mask] = rgb

    return color


def colorize_instance(instance: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Convert instance labels to color image.

    Args:
        instance: (H, W) instance IDs (0=background).
        semantic: (H, W) semantic labels.

    Returns:
        (H, W, 3) color image with random colors per instance.
    """
    h, w = instance.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    # Use semantic colors as base
    # color = colorize_semantic(semantic)

    # Override thing instances with random colors
    unique_instances = np.unique(instance)
    np.random.seed(42)  # For reproducibility

    for inst_id in unique_instances:
        if inst_id == 0:
            continue  # Skip background

        mask = instance == inst_id
        # Random color for each instance
        inst_color = np.random.randint(50, 255, size=3)
        color[mask] = inst_color

    return color


def visualize_predictions(
    image: np.ndarray,
    predictions: dict,
    gt_semantic: np.ndarray = None,
    save_path: str = None,
):
    """Visualize model predictions.

    Args:
        image: (H, W, 3) input image (0-1 range).
        predictions: Dict with 'semantic', 'instance', 'center', etc.
        gt_semantic: Optional ground truth semantic labels.
        save_path: Optional path to save figure.
    """
    semantic = predictions["semantic"].cpu().numpy()
    instance = predictions["instance"].cpu().numpy()
    center = predictions["center_heatmap"].cpu().numpy()

    # Number of rows
    n_rows = 3 if gt_semantic is not None else 2

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))

    # Row 1: Input, Semantic, Instance
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    semantic_color = colorize_semantic(semantic)
    axes[0, 1].imshow(semantic_color)
    axes[0, 1].set_title("Semantic Prediction")
    axes[0, 1].axis("off")

    instance_color = colorize_instance(instance, semantic)
    axes[0, 2].imshow(instance_color)
    axes[0, 2].set_title("Instance Prediction")
    axes[0, 2].axis("off")

    # Row 2: Center heatmaps (first 3 thing classes)
    for i in range(3):
        if i < center.shape[0]:
            axes[1, i].imshow(center[i], cmap="hot")
            axes[1, i].set_title(f"Center: {CLASS_NAMES[i]}")
        axes[1, i].axis("off")

    # Row 3: Ground truth (if available)
    if gt_semantic is not None:
        gt_color = colorize_semantic(gt_semantic)

        axes[2, 0].imshow(image)
        axes[2, 0].set_title("Input Image")
        axes[2, 0].axis("off")

        axes[2, 1].imshow(gt_color)
        axes[2, 1].set_title("Ground Truth Semantic")
        axes[2, 1].axis("off")

        # Overlay
        overlay = (0.5 * image + 0.5 * semantic_color / 255.0).clip(0, 1)
        axes[2, 2].imshow(overlay)
        axes[2, 2].set_title("Prediction Overlay")
        axes[2, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, default="data/cityscapes")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="outputs/visualize")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--center_threshold", type=float, default=0.001, help="Center detection threshold")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model_cfg = {
        "type": "PanopticSegmentor",
        "backbone": {
            "type": "ResNet",
            "depth": 34,
            "pretrained": True,
            "out_indices": [1, 2, 3, 4],
        },
        "neck": {
            "type": "BiFPN",
            "in_channels": [64, 128, 256, 512],
            "out_channels": 256,
            "num_layers": 3,
        },
        "context": {
            "type": "ContextAggregator",
            "in_channels": 256,
            "out_channels": 256,
            "num_levels": 4,
            "target_level": 1,
            "fusion": "concat",
        },
        "head": {
            "type": "PanopticHead",
            "in_channels": 256,
            "num_things_classes": NUM_THINGS,
            "num_stuff_classes": NUM_STUFF,
            "inner_channels": 128,
            "scale_factor": 8,
        },
    }
    model = build_model(model_cfg)

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "state_dict" in checkpoint:
            # PyTorch Lightning checkpoint - remove only first "model." prefix
            state_dict = {
                k.removeprefix("model."): v
                for k, v in checkpoint["state_dict"].items()
            }
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint loaded, using random initialization")

    model = model.to(device)
    model.eval()

    # Dataset (no augmentation for visualization)
    transforms = Compose([
        RandomCrop((512, 1024)),
        Normalize(),
    ])

    dataset = CityscapesDataset(
        root=args.data_root,
        split=args.split,
        transforms=transforms,
    )

    # Also load raw dataset for visualization
    raw_dataset = CityscapesDataset(
        root=args.data_root,
        split=args.split,
        transforms=Compose([RandomCrop((512, 1024))]),  # Only crop, no normalize
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize samples
    print(f"Visualizing {args.num_samples} samples...")

    for i in range(min(args.num_samples, len(dataset))):
        # Set same random seed for both datasets
        np.random.seed(i)
        sample = dataset[i]
        np.random.seed(i)
        raw_sample = raw_dataset[i]

        # Get raw image for visualization
        raw_image = raw_sample["image"].numpy().transpose(1, 2, 0)  # (H, W, 3)

        # Prepare input
        image = sample["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

        # Predict
        with torch.no_grad():
            predictions = model.predict(
                image,
                img_size=image.shape[-2:],
                center_threshold=args.center_threshold,
            )

        # Move to CPU and remove batch dimension
        predictions = {k: v[0] for k, v in predictions.items()}

        # Get ground truth
        gt_semantic = sample["sem_target"].numpy()

        # Visualize
        save_path = output_dir / f"sample_{i:04d}.png"
        visualize_predictions(
            image=raw_image,
            predictions=predictions,
            gt_semantic=gt_semantic,
            save_path=str(save_path),
        )

    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
