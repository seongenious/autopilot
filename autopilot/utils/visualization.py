"""Visualization utilities for autopilot perception tasks.

Provides functions to visualize:
    - 3D bounding boxes projected onto images
    - BEV (Bird's Eye View) map predictions
    - Depth predictions
    - GT vs Prediction comparisons
"""

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


# Color palette for different classes (BGR format for OpenCV)
CLASS_COLORS = {
    'car': (0, 255, 0),          # Green
    'truck': (0, 200, 0),        # Dark green
    'trailer': (0, 150, 0),      # Darker green
    'bus': (255, 255, 0),        # Cyan
    'construction_vehicle': (128, 128, 0),
    'bicycle': (255, 0, 255),    # Magenta
    'motorcycle': (200, 0, 200),
    'pedestrian': (0, 0, 255),   # Red
    'traffic_cone': (0, 165, 255),  # Orange
    'barrier': (128, 128, 128),  # Gray
    'Vehicle': (0, 255, 0),      # Waymo
    'Pedestrian': (0, 0, 255),
    'Cyclist': (255, 0, 255),
}

# Default color for unknown classes
DEFAULT_COLOR = (255, 255, 255)


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Get color for a class name."""
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def project_3d_to_2d(
    points_3d: np.ndarray,
    lidar2img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: 3D points in lidar/ego frame, shape (N, 3).
        lidar2img: Transformation matrix from lidar to image, shape (4, 4).

    Returns:
        Tuple of:
            - 2D image coordinates, shape (N, 2)
            - Valid mask for points in front of camera, shape (N,)
    """
    # Add homogeneous coordinate
    ones = np.ones((points_3d.shape[0], 1))
    points_3d_homo = np.concatenate([points_3d, ones], axis=1)  # (N, 4)

    # Project to image
    points_img = points_3d_homo @ lidar2img.T  # (N, 4)

    # Normalize by depth
    depth = points_img[:, 2]
    valid_mask = depth > 0.1  # Points in front of camera

    points_2d = np.zeros((points_3d.shape[0], 2))
    points_2d[valid_mask, 0] = points_img[valid_mask, 0] / depth[valid_mask]
    points_2d[valid_mask, 1] = points_img[valid_mask, 1] / depth[valid_mask]

    return points_2d, valid_mask


def get_3d_box_corners(
    center: np.ndarray,
    size: np.ndarray,
    rotation: float,
) -> np.ndarray:
    """Get 8 corners of a 3D bounding box.

    Args:
        center: Box center (x, y, z).
        size: Box dimensions (length, width, height).
        rotation: Yaw rotation in radians.

    Returns:
        8 corner points, shape (8, 3).
    """
    l, w, h = size / 2

    # 8 corners in box frame
    corners = np.array([
        [-l, -w, -h],
        [-l, -w,  h],
        [-l,  w, -h],
        [-l,  w,  h],
        [ l, -w, -h],
        [ l, -w,  h],
        [ l,  w, -h],
        [ l,  w,  h],
    ])

    # Rotation matrix (around z-axis)
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])

    # Rotate and translate
    corners = corners @ R.T + center

    return corners


def draw_3d_box_on_image(
    image: np.ndarray,
    corners_2d: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw a 3D bounding box on an image (in-place).

    Args:
        image: Input image (H, W, 3). Modified in-place.
        corners_2d: 2D projected corners, shape (8, 2).
        color: Line color in BGR.
        thickness: Line thickness.
    """
    corners = corners_2d.astype(np.int32)

    # Draw bottom face
    for i, j in [(0, 2), (2, 6), (6, 4), (4, 0)]:
        cv2.line(image, tuple(corners[i]), tuple(corners[j]), color, thickness)

    # Draw top face
    for i, j in [(1, 3), (3, 7), (7, 5), (5, 1)]:
        cv2.line(image, tuple(corners[i]), tuple(corners[j]), color, thickness)

    # Draw vertical edges
    for i, j in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        cv2.line(image, tuple(corners[i]), tuple(corners[j]), color, thickness)

    # Draw front face with different color (to show orientation)
    front_color = tuple(min(c + 50, 255) for c in color)
    for i, j in [(4, 5), (5, 7), (7, 6), (6, 4)]:
        cv2.line(image, tuple(corners[i]), tuple(corners[j]), front_color, thickness + 1)


def visualize_detection(
    image: np.ndarray,
    gt_boxes: Optional[Dict] = None,
    pred_boxes: Optional[Dict] = None,
    lidar2img: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Visualize 3D detection results on an image.

    Args:
        image: Input image (H, W, 3), RGB format.
        gt_boxes: Ground truth boxes dict with keys:
            - 'centers': (N, 3) box centers
            - 'sizes': (N, 3) box dimensions (l, w, h)
            - 'rotations': (N,) yaw angles
            - 'labels': (N,) class indices
        pred_boxes: Predicted boxes dict with same keys plus:
            - 'scores': (N,) confidence scores
        lidar2img: Transformation matrix (4, 4).
        class_names: List of class names.
        score_threshold: Minimum score for displaying predictions.

    Returns:
        Annotated image.
    """
    # Convert RGB to BGR for OpenCV
    vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = vis_image.shape[:2]

    def draw_boxes(boxes: Dict, is_gt: bool = True):
        if boxes is None or lidar2img is None:
            return

        centers = boxes.get('centers', np.array([]))
        sizes = boxes.get('sizes', np.array([]))
        rotations = boxes.get('rotations', np.array([]))
        labels = boxes.get('labels', np.array([]))
        scores = boxes.get('scores', np.ones(len(centers)))

        for i in range(len(centers)):
            if not is_gt and scores[i] < score_threshold:
                continue

            # Get 3D corners
            corners_3d = get_3d_box_corners(centers[i], sizes[i], rotations[i])

            # Project to 2D
            corners_2d, valid = project_3d_to_2d(corners_3d, lidar2img)

            # Skip if not enough corners visible
            if valid.sum() < 4:
                continue

            # Check if corners are within image bounds
            in_bounds = (
                (corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < w) &
                (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < h)
            )
            if (valid & in_bounds).sum() < 4:
                continue

            # Get color - GT uses class color, Pred uses red
            if is_gt:
                if class_names and labels[i] < len(class_names):
                    class_name = class_names[int(labels[i])]
                    color = get_class_color(class_name)
                else:
                    color = (0, 255, 0)  # Green for GT
            else:
                color = (0, 0, 255)  # Red for predictions (BGR)

            # Draw box
            draw_3d_box_on_image(vis_image, corners_2d, color, 2)

            # Add label
            label_pos = corners_2d[valid].min(axis=0).astype(int)
            label_text = f"{'GT' if is_gt else 'Pred'}"
            if class_names and labels[i] < len(class_names):
                label_text += f": {class_names[int(labels[i])]}"
            if not is_gt:
                label_text += f" ({scores[i]:.2f})"

            cv2.putText(
                vis_image, label_text,
                (max(0, label_pos[0]), max(15, label_pos[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

    # Draw GT boxes first (so predictions overlay)
    draw_boxes(gt_boxes, is_gt=True)
    draw_boxes(pred_boxes, is_gt=False)

    # Convert back to RGB
    return cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)


def visualize_bev_map(
    gt_map: Optional[np.ndarray] = None,
    pred_map: Optional[np.ndarray] = None,
    map_size: Tuple[int, int] = (200, 200),
    resolution: float = 0.5,
) -> np.ndarray:
    """Visualize BEV map prediction vs ground truth.

    Args:
        gt_map: Ground truth map, shape (H, W) or (C, H, W).
        pred_map: Predicted map, same shape as gt_map.
        map_size: Output visualization size (H, W).
        resolution: Meters per pixel.

    Returns:
        Side-by-side visualization image.
    """
    def normalize_map(m: np.ndarray) -> np.ndarray:
        if m is None:
            return np.zeros((*map_size, 3), dtype=np.uint8)

        # Handle multi-channel maps
        if m.ndim == 3:
            # Take argmax for class maps
            m = m.argmax(axis=0) if m.shape[0] < m.shape[-1] else m.argmax(axis=-1)

        # Normalize to 0-255
        m = m.astype(np.float32)
        if m.max() > m.min():
            m = (m - m.min()) / (m.max() - m.min())
        m = (m * 255).astype(np.uint8)

        # Resize if needed
        if m.shape[:2] != map_size:
            m = cv2.resize(m, (map_size[1], map_size[0]))

        # Convert to RGB
        return cv2.applyColorMap(m, cv2.COLORMAP_VIRIDIS)

    gt_vis = normalize_map(gt_map)
    pred_vis = normalize_map(pred_map)

    # Add labels
    cv2.putText(gt_vis, "Ground Truth", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(pred_vis, "Prediction", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Concatenate side by side
    return np.concatenate([gt_vis, pred_vis], axis=1)


def visualize_depth(
    gt_depth: Optional[np.ndarray] = None,
    pred_depth: Optional[np.ndarray] = None,
    max_depth: float = 80.0,
) -> np.ndarray:
    """Visualize depth prediction vs ground truth.

    Args:
        gt_depth: Ground truth depth map, shape (H, W).
        pred_depth: Predicted depth map, shape (H, W).
        max_depth: Maximum depth for normalization.

    Returns:
        Side-by-side visualization image.
    """
    def normalize_depth(d: np.ndarray) -> np.ndarray:
        if d is None:
            return np.zeros((256, 384, 3), dtype=np.uint8)

        # Clip and normalize
        d = np.clip(d, 0, max_depth)
        d = (d / max_depth * 255).astype(np.uint8)

        # Apply colormap
        return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)

    gt_vis = normalize_depth(gt_depth)
    pred_vis = normalize_depth(pred_depth)

    # Add labels
    cv2.putText(gt_vis, "GT Depth", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(pred_vis, "Pred Depth", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Calculate error map if both available
    if gt_depth is not None and pred_depth is not None:
        error = np.abs(gt_depth - pred_depth)
        error_vis = normalize_depth(error)
        cv2.putText(error_vis, "Error", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return np.concatenate([gt_vis, pred_vis, error_vis], axis=1)

    return np.concatenate([gt_vis, pred_vis], axis=1)


def create_multi_view_visualization(
    images: List[np.ndarray],
    camera_names: Optional[List[str]] = None,
    layout: Tuple[int, int] = (2, 3),
) -> np.ndarray:
    """Create a multi-view visualization grid.

    Args:
        images: List of images to display.
        camera_names: Optional camera names for labels.
        layout: Grid layout (rows, cols).

    Returns:
        Combined visualization image.
    """
    rows, cols = layout
    if len(images) == 0:
        return np.zeros((256, 384, 3), dtype=np.uint8)

    # Resize all images to same size
    target_h, target_w = images[0].shape[:2]
    resized = []
    for i, img in enumerate(images):
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h))

        # Add camera label
        if camera_names and i < len(camera_names):
            img = img.copy()
            cv2.putText(img, camera_names[i], (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        resized.append(img)

    # Pad with black images if needed
    while len(resized) < rows * cols:
        resized.append(np.zeros_like(resized[0]))

    # Create grid
    grid_rows = []
    for r in range(rows):
        row_images = resized[r * cols:(r + 1) * cols]
        grid_rows.append(np.concatenate(row_images, axis=1))

    return np.concatenate(grid_rows, axis=0)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to numpy array for visualization."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def visualize_bev_detection(
    gt_boxes: Optional[Dict] = None,
    pred_boxes: Optional[Dict] = None,
    point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    class_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
    canvas_size: Tuple[int, int] = (800, 800),
) -> np.ndarray:
    """Visualize 3D detection results in BEV (Bird's Eye View).

    Args:
        gt_boxes: Ground truth boxes dict with keys:
            - 'centers': (N, 3) box centers
            - 'sizes': (N, 3) box dimensions (l, w, h)
            - 'rotations': (N,) yaw angles
            - 'labels': (N,) class indices
        pred_boxes: Predicted boxes dict with same keys plus 'scores'.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        class_names: List of class names.
        score_threshold: Minimum score for displaying predictions.
        canvas_size: Output image size (H, W).

    Returns:
        BEV visualization image (H, W, 3) in BGR format.
    """
    h, w = canvas_size
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    x_max, y_max = point_cloud_range[3], point_cloud_range[4]

    # Create canvas (dark gray background)
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 40

    # Draw grid with axis labels
    grid_spacing = 10.0  # meters
    for x in np.arange(x_min, x_max + grid_spacing, grid_spacing):
        px = int((x - x_min) / (x_max - x_min) * w)
        cv2.line(canvas, (px, 0), (px, h), (60, 60, 60), 1)
        # X axis labels at bottom
        if abs(x) < 0.1 or abs(x - 50) < 0.1 or abs(x + 50) < 0.1 or abs(x % 20) < 0.1:
            label = f"{int(x)}"
            cv2.putText(canvas, label, (px - 10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    for y in np.arange(y_min, y_max + grid_spacing, grid_spacing):
        py = int((1 - (y - y_min) / (y_max - y_min)) * h)
        cv2.line(canvas, (0, py), (w, py), (60, 60, 60), 1)
        # Y axis labels on left side
        if abs(y) < 0.1 or abs(y - 50) < 0.1 or abs(y + 50) < 0.1 or abs(y % 20) < 0.1:
            label = f"{int(y)}"
            cv2.putText(canvas, label, (5, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Draw axis arrows and labels
    # X-axis (right)
    cv2.arrowedLine(canvas, (w // 2 + 20, h // 2), (w // 2 + 60, h // 2), (200, 200, 200), 2, tipLength=0.3)
    cv2.putText(canvas, "X", (w // 2 + 65, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    # Y-axis (forward/up in BEV)
    cv2.arrowedLine(canvas, (w // 2, h // 2 - 20), (w // 2, h // 2 - 60), (200, 200, 200), 2, tipLength=0.3)
    cv2.putText(canvas, "Y", (w // 2 + 5, h // 2 - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Draw ego vehicle (center)
    ego_x = int(w / 2)
    ego_y = int(h / 2)
    ego_size = 15
    cv2.rectangle(canvas, (ego_x - ego_size, ego_y - ego_size * 2),
                  (ego_x + ego_size, ego_y + ego_size), (100, 100, 255), -1)
    cv2.arrowedLine(canvas, (ego_x, ego_y), (ego_x, ego_y - ego_size * 3),
                    (100, 100, 255), 2, tipLength=0.3)

    def world_to_pixel(x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        px = int((x - x_min) / (x_max - x_min) * w)
        py = int((1 - (y - y_min) / (y_max - y_min)) * h)
        return px, py

    def draw_rotated_box(center: np.ndarray, size: np.ndarray, rotation: float,
                         color: Tuple[int, int, int], thickness: int = 2):
        """Draw a rotated box in BEV."""
        l, wid = size[0], size[1]  # length and width
        corners = np.array([
            [-l/2, -wid/2],
            [l/2, -wid/2],
            [l/2, wid/2],
            [-l/2, wid/2],
        ])

        # Rotate
        c, s = np.cos(rotation), np.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        corners = corners @ R.T + center[:2]

        # Convert to pixels
        pts = np.array([world_to_pixel(c[0], c[1]) for c in corners], dtype=np.int32)

        # Draw box
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw direction indicator (front edge)
        front_center = (pts[1] + pts[2]) // 2
        cv2.circle(canvas, tuple(front_center), 4, color, -1)

    def draw_boxes(boxes: Dict, is_gt: bool = True):
        if boxes is None:
            return

        centers = boxes.get('centers', np.array([]))
        sizes = boxes.get('sizes', np.array([]))
        rotations = boxes.get('rotations', np.array([]))
        labels = boxes.get('labels', np.array([]))
        scores = boxes.get('scores', np.ones(len(centers)))

        if len(centers) == 0:
            return

        for i in range(len(centers)):
            if not is_gt and scores[i] < score_threshold:
                continue

            # Get color - GT is green, Pred is red/orange
            if is_gt:
                if class_names and int(labels[i]) < len(class_names):
                    class_name = class_names[int(labels[i])]
                    color = get_class_color(class_name)
                else:
                    color = (0, 255, 0)  # Green for GT
            else:
                # Use distinct red/orange colors for predictions
                color = (0, 0, 255)  # Red for predictions (BGR)

            thickness = 2
            draw_rotated_box(centers[i], sizes[i], rotations[i], color, thickness)

    # Draw GT first, then predictions
    draw_boxes(gt_boxes, is_gt=True)
    draw_boxes(pred_boxes, is_gt=False)

    # Add legend
    legend_y = 30
    cv2.putText(canvas, "BEV View", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    legend_y += 30
    cv2.rectangle(canvas, (10, legend_y - 10), (25, legend_y + 5), (0, 255, 0), -1)
    cv2.putText(canvas, "GT", (30, legend_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 25
    cv2.rectangle(canvas, (10, legend_y - 10), (25, legend_y + 5), (0, 0, 255), -1)
    cv2.putText(canvas, "Pred", (30, legend_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add scale bar
    scale_length = 20  # meters
    scale_px = int(scale_length / (x_max - x_min) * w)
    cv2.line(canvas, (w - 20 - scale_px, h - 30), (w - 20, h - 30), (255, 255, 255), 2)
    cv2.putText(canvas, f"{scale_length}m", (w - 20 - scale_px // 2 - 15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return canvas


def create_evaluation_figure(
    images: List[np.ndarray],
    gt_boxes: Optional[Dict] = None,
    pred_boxes: Optional[Dict] = None,
    lidar2img: Optional[np.ndarray] = None,
    point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    class_names: Optional[List[str]] = None,
    camera_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Create evaluation figure with camera views and BEV.

    Left side: 6 camera images (2 rows x 3 cols) with detection overlay
    Right side: BEV visualization with GT and predictions

    Args:
        images: List of 6 camera images (H, W, 3), RGB format.
        gt_boxes: Ground truth boxes dictionary.
        pred_boxes: Predicted boxes dictionary.
        lidar2img: Transformation matrices (num_cam, 4, 4).
        point_cloud_range: BEV range [x_min, y_min, z_min, x_max, y_max, z_max].
        class_names: List of class names.
        camera_names: List of camera names.
        score_threshold: Minimum score threshold.

    Returns:
        Combined figure image (H, W, 3) in RGB format.
    """
    if camera_names is None:
        camera_names = ['CAM_FL', 'CAM_F', 'CAM_FR', 'CAM_BL', 'CAM_B', 'CAM_BR']

    # Process each camera image with detection overlay
    vis_images = []
    for cam_idx, img in enumerate(images):
        cam_lidar2img = lidar2img[cam_idx] if lidar2img is not None else None

        vis_img = visualize_detection(
            img,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            lidar2img=cam_lidar2img,
            class_names=class_names,
            score_threshold=score_threshold,
        )
        vis_images.append(vis_img)

    # Create camera grid (2 rows x 3 cols)
    # Resize images for consistent grid
    target_h, target_w = images[0].shape[:2]
    resized = []
    for i, img in enumerate(vis_images):
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h))

        # Add camera label
        img = img.copy()
        label = camera_names[i] if i < len(camera_names) else f"CAM_{i}"
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        resized.append(img)

    # Arrange in 2x3 grid
    row1 = np.concatenate(resized[0:3], axis=1)  # Front left, Front, Front right
    row2 = np.concatenate(resized[3:6], axis=1)  # Back left, Back, Back right
    camera_grid = np.concatenate([row1, row2], axis=0)

    # Create BEV visualization
    bev_canvas_size = (camera_grid.shape[0], camera_grid.shape[0])  # Square, same height as camera grid
    bev_vis = visualize_bev_detection(
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        point_cloud_range=point_cloud_range,
        class_names=class_names,
        score_threshold=score_threshold,
        canvas_size=bev_canvas_size,
    )

    # Convert BEV from BGR to RGB
    bev_vis = cv2.cvtColor(bev_vis, cv2.COLOR_BGR2RGB)

    # Combine camera grid and BEV side by side
    combined = np.concatenate([camera_grid, bev_vis], axis=1)

    return combined


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
    std: Tuple[float, ...] = (58.395, 57.12, 57.375),
) -> np.ndarray:
    """Denormalize an image for visualization.

    Args:
        image: Normalized image, shape (C, H, W) or (H, W, C).
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Denormalized image in uint8, shape (H, W, C).
    """
    if image.shape[0] == 3:  # CHW -> HWC
        image = image.transpose(1, 2, 0)

    mean = np.array(mean)
    std = np.array(std)

    image = image * std + mean
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image
