"""NuScenes dataset converter.

Converts NuScenes dataset to pickle format for training.
"""

import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.utils.geometry_utils import transform_matrix
except ImportError:
    raise ImportError("Please install nuscenes-devkit: pip install nuscenes-devkit")


# NuScenes class names for detection
NUSCENES_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# Camera names
CAMERA_NAMES = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
]


def get_available_scenes(nusc: NuScenes) -> List[Dict]:
    """Get all available scenes."""
    available_scenes = []
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])

        # Check if the scene has valid data
        has_valid_data = True
        for cam in CAMERA_NAMES:
            if cam not in sample_rec['data']:
                has_valid_data = False
                break

        if has_valid_data:
            available_scenes.append(scene)

    return available_scenes


def check_sample_images_exist(nusc: NuScenes, sample_token: str, data_root: str) -> bool:
    """Check if all camera images exist for a sample.

    Args:
        nusc: NuScenes instance.
        sample_token: Sample token.
        data_root: Path to NuScenes data root.

    Returns:
        True if all camera images exist, False otherwise.
    """
    sample = nusc.get('sample', sample_token)

    for cam_name in CAMERA_NAMES:
        if cam_name not in sample['data']:
            return False

        cam_token = sample['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        img_path = os.path.join(data_root, cam_data['filename'])

        if not os.path.exists(img_path):
            return False

    return True


def get_sample_data(
    nusc: NuScenes,
    sample_token: str,
    min_visibility: int = 3,
) -> Dict:
    """Get data for a single sample.

    Args:
        nusc: NuScenes instance.
        sample_token: Sample token.
        min_visibility: Minimum visibility level (1-4).

    Returns:
        Dictionary containing sample data.
    """
    sample = nusc.get('sample', sample_token)

    # Get lidar data for reference frame
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)

    # Get ego pose
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])

    # Get lidar calibration
    lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    lidar_translation = np.array(lidar_calib['translation'])
    lidar_rotation = Quaternion(lidar_calib['rotation'])

    # Camera data
    cam_infos = {}
    for cam_name in CAMERA_NAMES:
        cam_token = sample['data'][cam_name]
        cam_data = nusc.get('sample_data', cam_token)
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])

        # Camera intrinsics
        intrinsics = np.array(cam_calib['camera_intrinsic'])

        # Camera extrinsics (camera to ego)
        cam_translation = np.array(cam_calib['translation'])
        cam_rotation = Quaternion(cam_calib['rotation'])

        # Ego to global
        ego_trans = np.array(cam_ego_pose['translation'])
        ego_rot = Quaternion(cam_ego_pose['rotation'])

        # Build transformation matrices
        # sensor2ego
        sensor2ego = transform_matrix(cam_translation, cam_rotation, inverse=False)
        # ego2global
        ego2global = transform_matrix(ego_trans, ego_rot, inverse=False)
        # sensor2global = ego2global @ sensor2ego
        sensor2global = ego2global @ sensor2ego

        # For lidar2cam, we need: global2sensor @ ego2lidar @ lidar2ego @ ego2global
        # Simplified: we compute lidar2img directly

        # lidar2ego
        lidar2ego = transform_matrix(lidar_translation, lidar_rotation, inverse=False)
        # ego2lidar
        ego2lidar = transform_matrix(lidar_translation, lidar_rotation, inverse=True)

        # Build lidar2cam
        # cam2ego
        cam2ego = sensor2ego
        # ego2cam
        ego2cam = transform_matrix(cam_translation, cam_rotation, inverse=True)

        # lidar2cam = ego2cam @ lidar2ego
        lidar2cam = ego2cam @ lidar2ego

        # lidar2img
        viewpad = np.eye(4)
        viewpad[:3, :3] = intrinsics
        lidar2img = viewpad @ lidar2cam

        cam_infos[cam_name] = {
            'filename': cam_data['filename'],
            'intrinsics': intrinsics,
            'extrinsics': lidar2cam[:3, :],  # 3x4 matrix
            'lidar2img': lidar2img,
            'sensor2ego': sensor2ego,
            'ego2global': ego2global,
            'timestamp': cam_data['timestamp'],
        }

    # Get annotations (3D bounding boxes)
    annotations = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        # Get visibility
        visibility_token = ann['visibility_token']
        visibility_info = nusc.get('visibility', visibility_token)
        # Visibility level is stored as 'v0-40', 'v40-60', 'v60-80', 'v80-100'
        # Map to levels 1, 2, 3, 4
        visibility_map = {'v0-40': 1, 'v40-60': 2, 'v60-80': 3, 'v80-100': 4}
        visibility_level = visibility_info.get('level', visibility_info.get('token', ''))
        visibility = visibility_map.get(visibility_level, 1)

        # Filter by visibility
        if visibility < min_visibility:
            continue

        # Get category
        category = ann['category_name']

        # Map to our class names
        class_name = None
        for cls in NUSCENES_CLASSES:
            if cls in category:
                class_name = cls
                break

        if class_name is None:
            continue

        # Get box parameters in global frame
        center = np.array(ann['translation'])
        size = np.array(ann['size'])  # [w, l, h] in nuscenes
        rotation = Quaternion(ann['rotation'])

        # Convert to lidar frame
        # global2ego
        global2ego = transform_matrix(ego_translation, ego_rotation, inverse=True)
        # ego2lidar
        ego2lidar = transform_matrix(lidar_translation, lidar_rotation, inverse=True)
        # global2lidar
        global2lidar = ego2lidar @ global2ego

        # Transform center
        center_homo = np.array([*center, 1.0])
        center_lidar = (global2lidar @ center_homo)[:3]

        # Transform rotation
        # Get yaw angle in lidar frame
        global_yaw = rotation.yaw_pitch_roll[0]
        ego_yaw = ego_rotation.yaw_pitch_roll[0]
        lidar_yaw = lidar_rotation.yaw_pitch_roll[0]
        box_yaw = global_yaw - ego_yaw - lidar_yaw

        # NuScenes uses [w, l, h], we use [l, w, h]
        # x, y, z, l, w, h, yaw
        box_3d = np.array([
            center_lidar[0],  # x
            center_lidar[1],  # y
            center_lidar[2],  # z
            size[1],          # length (nuscenes l)
            size[0],          # width (nuscenes w)
            size[2],          # height
            box_yaw,          # yaw
        ])

        annotations.append({
            'class_name': class_name,
            'class_id': NUSCENES_CLASSES.index(class_name),
            'box_3d': box_3d,  # [x, y, z, l, w, h, yaw]
            'num_lidar_pts': ann['num_lidar_pts'],
            'num_radar_pts': ann['num_radar_pts'],
            'visibility': visibility,
            'instance_token': ann['instance_token'],
        })

    # Get scene info
    scene = nusc.get('scene', sample['scene_token'])

    # Build info dict
    info = {
        'token': sample_token,
        'scene_token': sample['scene_token'],
        'scene_name': scene['name'],
        'timestamp': sample['timestamp'],
        'lidar_path': lidar_data['filename'],
        'sweeps': [],  # Can be extended for multi-sweep
        'cams': cam_infos,
        'lidar2ego_translation': lidar_translation,
        'lidar2ego_rotation': lidar_rotation.elements,
        'ego2global_translation': ego_translation,
        'ego2global_rotation': ego_rotation.elements,
        'gt_boxes': np.array([ann['box_3d'] for ann in annotations]) if annotations else np.zeros((0, 7)),
        'gt_names': np.array([ann['class_name'] for ann in annotations]) if annotations else np.array([]),
        'gt_labels': np.array([ann['class_id'] for ann in annotations]) if annotations else np.zeros((0,), dtype=np.int64),
        'num_lidar_pts': np.array([ann['num_lidar_pts'] for ann in annotations]) if annotations else np.zeros((0,), dtype=np.int64),
        'visibility': np.array([ann['visibility'] for ann in annotations]) if annotations else np.zeros((0,), dtype=np.int64),
    }

    return info


def create_nuscenes_infos(
    root_path: str,
    version: str = 'v1.0-trainval',
    max_sweeps: int = 10,
    min_visibility: int = 3,
) -> Tuple[List[Dict], List[Dict]]:
    """Create NuScenes info files.

    Args:
        root_path: Path to NuScenes data root.
        version: NuScenes version.
        max_sweeps: Maximum number of sweeps (not used currently).
        min_visibility: Minimum visibility level for annotations.

    Returns:
        Tuple of (train_infos, val_infos).
    """
    print(f"Loading NuScenes {version}...")
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    # Get scene splits
    if version == 'v1.0-mini':
        train_scenes = create_splits_scenes()['mini_train']
        val_scenes = create_splits_scenes()['mini_val']
    else:
        train_scenes = create_splits_scenes()['train']
        val_scenes = create_splits_scenes()['val']

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]

    train_scenes = [s for s in train_scenes if s in available_scene_names]
    val_scenes = [s for s in val_scenes if s in available_scene_names]

    print(f"Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}")

    # Process train samples
    train_infos = []
    train_skipped = 0
    print("Processing train samples...")
    for scene in tqdm(available_scenes):
        if scene['name'] not in train_scenes:
            continue

        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            # Check if all camera images exist
            if check_sample_images_exist(nusc, sample_token, root_path):
                info = get_sample_data(nusc, sample_token, min_visibility)
                train_infos.append(info)
            else:
                train_skipped += 1
            sample_token = sample['next']

    # Process val samples
    val_infos = []
    val_skipped = 0
    print("Processing val samples...")
    for scene in tqdm(available_scenes):
        if scene['name'] not in val_scenes:
            continue

        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            # Check if all camera images exist
            if check_sample_images_exist(nusc, sample_token, root_path):
                info = get_sample_data(nusc, sample_token, min_visibility)
                val_infos.append(info)
            else:
                val_skipped += 1
            sample_token = sample['next']

    print(f"Train samples: {len(train_infos)} (skipped {train_skipped} with missing images)")
    print(f"Val samples: {len(val_infos)} (skipped {val_skipped} with missing images)")

    return train_infos, val_infos


def main():
    parser = argparse.ArgumentParser(description='Create NuScenes info files')
    parser.add_argument('--root-path', type=str, default='data/nuscenes',
                        help='Path to NuScenes data root')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                        help='NuScenes version')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: same as root-path)')
    parser.add_argument('--max-sweeps', type=int, default=10,
                        help='Maximum number of sweeps')
    parser.add_argument('--min-visibility', type=int, default=1,
                        help='Minimum visibility level (1-4)')
    args = parser.parse_args()

    out_dir = args.out_dir or args.root_path
    os.makedirs(out_dir, exist_ok=True)

    train_infos, val_infos = create_nuscenes_infos(
        args.root_path,
        args.version,
        args.max_sweeps,
        args.min_visibility,
    )

    # Save to pickle files
    if args.version == 'v1.0-mini':
        train_file = os.path.join(out_dir, 'nuscenes_infos_train_mini.pkl')
        val_file = os.path.join(out_dir, 'nuscenes_infos_val_mini.pkl')
    else:
        train_file = os.path.join(out_dir, 'nuscenes_infos_train.pkl')
        val_file = os.path.join(out_dir, 'nuscenes_infos_val.pkl')

    print(f"Saving train infos to {train_file}")
    with open(train_file, 'wb') as f:
        pickle.dump(train_infos, f)

    print(f"Saving val infos to {val_file}")
    with open(val_file, 'wb') as f:
        pickle.dump(val_infos, f)

    print("Done!")


if __name__ == '__main__':
    main()
