"""
KITTI LiDAR-Camera Fusion for Ground Plane Segmentation
Traditional Computer Vision Approach (No ML)

Dataset: KITTI Raw Data
URL: https://www.cvlibs.net/datasets/kitti/raw_data.php
"""

import cv2
import numpy as np
import fusion_utils as fu
from kitti_dataloader import KITTIDataLoader
from ground_plane_segmentation import GroundPlaneSegmentation
from tqdm import tqdm # pyright: ignore[reportMissingModuleSource]


"""
    Fuse LiDAR and camera segmentations
    
    Args:
        lidar_mask: LiDAR ground segmentation mask
        camera_mask: Camera road segmentation mask
        lidar_weight: Weight for LiDAR evidence (0-1)
        
    Returns:
        fused_mask: Combined segmentation mask
"""
def fuse_segmentations(lidar_mask: np.ndarray,
                        camera_mask: np.ndarray) -> np.ndarray:
    # Normalize masks
    lidar_evidence = lidar_mask.astype(np.float32) / 255.0
    camera_evidence = camera_mask.astype(np.float32) / 255.0

    # Apply Gaussian smoothing to LiDAR mask for better coverage
    lidar_evidence = cv2.GaussianBlur(lidar_evidence, (21, 21), 0)
    
    # Apply a smooth weighting based on the distance to the ego vehicle
    H = camera_evidence.shape[0]
    y = np.arange(H).reshape(H, 1)
    lidar_weight_map = y / (H - 1)
    lidar_weight_map = np.repeat(lidar_weight_map, camera_evidence.shape[1], axis=1)
    
    # Normalize after blur
    if lidar_evidence.max() > 0:
        lidar_evidence = lidar_evidence / lidar_evidence.max()
    
    # Weighted fusion
    fused = lidar_weight_map * lidar_evidence + (1 - lidar_weight_map) * camera_evidence
    
    # Threshold
    fused_mask = (fused > 0.4).astype(np.uint8) * 255
    
    # Clean up with morphology
    kernel = np.ones((7, 7), np.uint8)
    fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_CLOSE, kernel)
    fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
    
    return fused_mask

"""
    Start point of execution of the pipeline for KITTI dataset
"""
def main():
    # Configurable constant to point at the KITTI dataset
    BASE_PATH = r'./kitti_raw'  # first folder of the dataset
    DATE = '2011_09_26'  # Date folder
    DRIVE = '0005'  # Drive number (with leading zeros)
    CAMERA = 'image_02' # From which camera take the pictures
    FRAME_IDX = 110  # Which frame to process
    IMAGE_NET_PATH = './U_Net'
    
    print("="*60)
    print("KITTI Ground Plane Segmentation Pipeline")
    print("="*60)
    
    # Load KITTI dataset
    print("\n1. Loading KITTI dataset...")
    try:
        kitti = KITTIDataLoader(BASE_PATH, DATE, DRIVE, CAMERA)
    except Exception as e:
        print(f"\nError loading KITTI dataset: {e}")
        print("\nPlease update the configuration:")
        print("  BASE_PATH: Path to KITTI raw data root")
        print("  DATE: Date folder (e.g., '2011_09_26')")
        print("  DRIVE: Drive number (e.g., '0001')")
        return
    
    # Initialize segmentation
    segmenter = GroundPlaneSegmentation(kitti, IMAGE_NET_PATH)
    
    # Load frame
    print(f"\n2. Loading frame {FRAME_IDX}...")
    frame = kitti.get_frame(FRAME_IDX)
    print(f"   Image shape: {frame['image'].shape}")
    print(f"   Point cloud size: {frame['points'].shape}")
    
    # Project to camera
    print("\n3. Projecting Velodyne points to camera...")
    valid_points, pixel_coords, depths, valid_indices = segmenter.project_velodyne_to_camera(
        frame['points'],
        frame['calib'],
        frame['image'].shape[:2]
    )
    print(f"   Valid projected points: {len(valid_points)}")
    print(f"   Depth range: {depths.min():.1f}m - {depths.max():.1f}m")
    
    # Segment ground from LiDAR (External RANSAC)
    print("\n4. Segmenting ground plane from LiDAR (RANSAC)...")
    ground_mask = segmenter.segment_ground_lidar(frame['points'])
    ground_points = np.sum(ground_mask)
    print(f"   Ground points: {ground_points}/{len(ground_mask)} ({100*ground_points/len(ground_mask):.1f}%)")
    
    # Match ground mask with projected points using pre-computed valid indices
    print("\n5. Mapping ground mask to projected points...")
    ground_mask_projected = ground_mask[valid_indices]
    print(f"   Projected ground points: {np.sum(ground_mask_projected)}/{len(ground_mask_projected)} ({100*np.sum(ground_mask_projected)/len(ground_mask_projected):.1f}%)")
   
    # Create LiDAR mask image
    print("\n6. Creating LiDAR ground mask image...")
    lidar_mask = segmenter.create_lidar_ground_mask_image(
        frame['image'].shape[:2],
        pixel_coords,
        ground_mask_projected
    )
    
    # Segment road from camera
    print("\n7. Segmenting road from camera")
    camera_mask = segmenter.segment_road_camera(frame['image'])
    
    # Fuse segmentations
    print("\n8. Fusing LiDAR and camera segmentations...")
    fused_mask = fuse_segmentations(
        lidar_mask,
        camera_mask
    )

    fu.visualize_comparison(frame['image'], lidar_mask, camera_mask, fused_mask, pixel_coords, depths)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
