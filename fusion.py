"""
KITTI LiDAR-Camera Fusion for Ground Plane Segmentation
Traditional Computer Vision Approach (No ML)

Dataset: KITTI Raw Data
URL: https://www.cvlibs.net/datasets/kitti/raw_data.php
"""

import numpy as np
import cv2
import os
import glob
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm # pyright: ignore[reportMissingModuleSource]
import matplotlib as mpl
import torch
from U_Net.U_Net import UNet
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from RANSAC.ransac import RANSAC

"""
    Class to load and parse KITTI raw dataset
"""
class KITTIDataLoader:
    """
        Initialize KITTI dataset loader
        
        Args:
            base_path: Path to KITTI raw data root
            date: Date folder (e.g., '2011_09_26')
            drive: Drive folder (e.g., '0001')
    """
    def __init__(self, base_path: str, date: str, drive: str, camera: str):
        self.base_path = base_path
        self.date = date
        self.drive = drive
        self.camera = camera
        
        # Construct paths
        self.drive_path = os.path.join(base_path, date, f"{date}_drive_{drive}_sync")
        self.calib_path = os.path.join(base_path, date)
        
        # Verify paths exist
        if not os.path.exists(self.drive_path):
            raise ValueError(f"Drive path does not exist: {self.drive_path}")
        if not os.path.exists(self.calib_path):
            raise ValueError(f"Calibration path does not exist: {self.calib_path}")
        
        # Load calibration data
        self.calib = self.load_calibration()
        
        # Get file lists
        self.image_files = sorted(glob.glob(os.path.join(
            self.drive_path, self.camera, "data", "*.png")))
        self.velodyne_files = sorted(glob.glob(os.path.join(
            self.drive_path, "velodyne_points", "data", "*.bin")))
        
        print(f"Loaded KITTI dataset:")
        print(f"  Date: {date}")
        print(f"  Drive: {drive}")
        print(f"  Images: {len(self.image_files)}")
        print(f"  Velodyne scans: {len(self.velodyne_files)}")
    

    """
        Load calibration files from KITTI
        
        Returns:
            Dictionary with calibration matrices
    """
    def load_calibration(self) -> Dict:
        calib = {}
        
        # Load camera-to-camera calibration (calib_cam_to_cam.txt)
        cam_to_cam_file = os.path.join(self.calib_path, "calib_cam_to_cam.txt")
        with open(cam_to_cam_file, 'r') as f:
            # skip first line as it contains timestamp data
            for line in f.readlines()[1:]:
                if line.strip():
                    key, value = line.split(':', 1)
                    key = key.strip()
                    # Parse matrix values
                    if 'R_' in key or 'T_' in key or 'S_' in key or 'P_' in key or 'K_' in key or 'D_' in key:
                        calib[key] = np.array([float(x) for x in value.split()])
        
        # Load velo-to-camera calibration (calib_velo_to_cam.txt)
        velo_to_cam_file = os.path.join(self.calib_path, "calib_velo_to_cam.txt")
        with open(velo_to_cam_file, 'r') as f:
            # skip first line as it contains timestamp data
            for line in f.readlines()[1:]:
                if line.strip():
                    key, value = line.split(':', 1)
                    key = key.strip()
                    calib[key] = np.array([float(x) for x in value.split()])
        
        # Reshape matrices
        # LiDAR calibration parameters with respect to camera 0
        if 'R' in calib:
            calib['R'] = calib['R'].reshape(3, 3)
        if 'T' in calib:
            calib['T'] = calib['T'].reshape(3, 1)
        
        # Camera 2 (left color camera) projection matrix
        if 'P_rect_02' in calib:
            calib['P2'] = calib['P_rect_02'].reshape(3, 4)
        
        # Rectification matrix for camera 0
        if 'R_rect_00' in calib:
            calib['R_rect'] = calib['R_rect_00'].reshape(3, 3)
        
        return calib
    
    
    """
        Load Velodyne point cloud
        
        Args:
            idx: Frame index
            
        Returns:
            Points (N, 4) with [x, y, z, reflectance]
    """
    def load_velodyne(self, idx: int) -> np.ndarray:
        points = np.fromfile(self.velodyne_files[idx], dtype=np.float32).reshape(-1, 4)
        return points
    

    """
        Load camera image
        
        Args:
            idx: Frame index
            
        Returns:
            RGB image
    """
    def load_image(self, idx: int) -> np.ndarray:
        image = cv2.imread(self.image_files[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    

    """
        Get synchronized frame data
        
        Args:
            idx: Frame index
            
        Returns:
            Dictionary with image, point cloud, and calibration
    """
    def get_frame(self, idx: int) -> Dict:
        if idx >= len(self.image_files) or idx >= len(self.velodyne_files):
            raise ValueError(f"Frame index {idx} out of range")
        
        return {
            'image': self.load_image(idx),
            'points': self.load_velodyne(idx),
            'calib': self.calib,
            'idx': idx
        }


"""
    Class for ground plane segmentation using LiDAR-Camera fusion
"""
class GroundPlaneSegmentation:
    """
        Initialize segmentation pipeline
        
        Args:
            kitti_loader: KITTI data loader instance
    """
    def __init__(self, kitti_loader: KITTIDataLoader, image_net_path: str):
        self.loader = kitti_loader
        self.image_net_path = image_net_path

        # Network definition
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = UNet().to(self.device)

        # Trained weights loading
        pth_files = [f for f in os.listdir(self.image_net_path) if f.endswith(".pth")]
        if not pth_files:
            raise FileNotFoundError(f"No trained weights found in {image_net_path}")
        path = os.path.join(image_net_path, pth_files[0])

        self.net.load_state_dict(torch.load(path, self.device))
        
        # Putting the model in evaluation mode
        self.net.eval()


    """
        Project Velodyne points to camera image plane
        
        Args:
            points: Velodyne points (N, 4) with [x, y, z, reflectance]
            calib: Calibration dictionary
            image_shape: (height, width) of image
            
        Returns:
            valid_points: Original 3D points in Velodyne frame (M, 3)
            pixel_coords: Pixel coordinates (M, 2) [u, v]
            depths: Depth values (M,)
            valid_indices: Original indices of valid points in input array (M,)
    """
    def project_velodyne_to_camera(self, points: np.ndarray, calib: Dict,
                                   image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Extract xyz
        pts_3d = points[:, :3]  # (N, 3)
        
        # Create homogeneous coordinates
        pts_3d_hom = np.hstack([pts_3d, np.ones((pts_3d.shape[0], 1))])  # (N, 4)
        
        # Velodyne to camera 0 transformation
        R = calib['R']  # (3, 3)
        T = calib['T']  # (3, 1)
        
        # Create 4x4 transformation matrix from LiDAR to camera 0
        RT = np.eye(4)
        RT[:3, :3] = R
        RT[:3, 3:4] = T
        
        # Transform to camera 0 coordinates
        pts_cam0 = RT @ pts_3d_hom.T  # (4, N)
        pts_cam0 = pts_cam0[:3, :].T  # (N, 3)
        
        # Apply rectification
        R_rect = np.eye(4)
        R_rect[:3, :3] = calib['R_rect']
        pts_cam0_hom = np.hstack([pts_cam0, np.ones((pts_cam0.shape[0], 1))])
        pts_rect = R_rect @ pts_cam0_hom.T  # (4, N)
        pts_rect = pts_rect[:3, :].T  # (N, 3)
        
        # Filter points behind camera
        behind_mask = pts_rect[:, 2] > 0
        pts_rect_filtered = pts_rect[behind_mask]
        indices_after_behind = np.where(behind_mask)[0]
        
        # Project to image plane using P2 (camera 2 - left color)
        P2 = calib['P2']  # (3, 4)
        pts_rect_hom = np.hstack([pts_rect_filtered, np.ones((pts_rect_filtered.shape[0], 1))])
        pts_2d = P2 @ pts_rect_hom.T  # (3, M)
        
        # Normalize by depth
        pts_2d = pts_2d / pts_2d[2:3, :]
        
        # Get pixel coordinates
        pixel_coords = pts_2d[:2, :].T  # (M, 2) [u, v]
        depths = pts_rect_filtered[:, 2]  # (M,)
        
        # Filter points outside image bounds
        height, width = image_shape
        valid_mask = (
            (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < width) &
            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < height)
        )
        
        # Get final valid indices in original array
        valid_indices = indices_after_behind[valid_mask]
        
        # Return original Velodyne coordinates (not rectified)
        valid_points = pts_3d[valid_indices]
        pixel_coords = pixel_coords[valid_mask]
        depths = depths[valid_mask]
        
        return valid_points, pixel_coords, depths, valid_indices
    
    
    def temp_segment_lidar(self, points: np.ndarray,
                            distance_threshold: float = 0.2,
                            ransac_n: int = 3,
                            num_iterations: int = 100) -> np.ndarray:
        ransac = RANSAC(points, num_iterations, distance_threshold)

        inlier_points = ransac._ransac_algorithm(num_iterations, distance_threshold)

        return inlier_points
    


    """
        Segment ground plane from LiDAR using RANSAC
        
        Args:
            points: LiDAR points (N, 3 or 4)
            distance_threshold: RANSAC inlier threshold
            ransac_n: Number of points to sample
            num_iterations: RANSAC iterations
            
        Returns:
            ground_mask: Boolean mask for ground points
    """
    def segment_ground_lidar(self, points: np.ndarray,
                            distance_threshold: float = 0.2,
                            ransac_n: int = 3,
                            num_iterations: int = 100) -> np.ndarray:
        xyz = points[:, :3]
        best_inliers = []
        
        for _ in range(num_iterations):
            # Randomly sample points
            sample_idx = np.random.choice(len(xyz), ransac_n, replace=False)
            sample_points = xyz[sample_idx]
            
            # Fit plane using SVD
            centroid = np.mean(sample_points, axis=0)
            centered = sample_points - centroid
            _, _, vh = np.linalg.svd(centered)
            normal = vh[2, :]  # Last row is normal vector
            
            # Ensure normal points upward (negative z in KITTI coordinates)
            if normal[2] > 0:
                normal = -normal
            
            # Calculate distances to plane
            distances = np.abs(np.dot(xyz - centroid, normal))
            inliers = distances < distance_threshold
            
            # Update best plane
            if np.sum(inliers) > np.sum(best_inliers):
                best_inliers = inliers
        
        return best_inliers


    """
        Segment road from camera image using a neural network
        
        Args:
            image: RGB image
            
        Returns:
            road_mask: Binary mask for road pixels
    """
    def segment_road_camera(self, image: np.ndarray) -> np.ndarray:
        # Image transformation
        image_transforms = transforms.Compose([
            transforms.Resize((128, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[0:3])
        ])

        image = Image.fromarray(image)
        image = image_transforms(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = self.net(image)
        output = torch.argmax(output, dim=1).detach().squeeze(0).float().unsqueeze(0).unsqueeze(0)

        # Interpolation to resize the prediction to the original image shape
        output = F.interpolate(output, size=(375, 1242), mode='nearest')

        # Keeping only the ground classes as mask
        ground_classes = torch.tensor((0,1,9))
        road_mask = torch.isin(output, ground_classes).squeeze().cpu().to(torch.uint8)

        return torch.Tensor.numpy(road_mask)
    

    """
        Create binary mask image from projected ground points
        
        Args:
            image_shape: (height, width)
            pixel_coords: Pixel coordinates (M, 2)
            ground_mask_projected: Ground mask for projected points (M,)
            
        Returns:
            Binary mask image
    """
    def create_lidar_ground_mask_image(self, image_shape: Tuple[int, int],
                                       pixel_coords: np.ndarray,
                                       ground_mask_projected: np.ndarray) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        ground_pixels = pixel_coords[ground_mask_projected]
        for pt in ground_pixels:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                cv2.circle(mask, (x, y), 5, 255, -1)
        
        return mask
    

    """
        Fuse LiDAR and camera segmentations
        
        Args:
            lidar_mask: LiDAR ground segmentation mask
            camera_mask: Camera road segmentation mask
            lidar_weight: Weight for LiDAR evidence (0-1)
            
        Returns:
            fused_mask: Combined segmentation mask
    """
    def fuse_segmentations(self, lidar_mask: np.ndarray,
                          camera_mask: np.ndarray,
                          lidar_weight: float = 0.6) -> np.ndarray:
        # Normalize masks
        lidar_evidence = lidar_mask.astype(np.float32) / 255.0
        camera_evidence = camera_mask.astype(np.float32) / 255.0
        
        # Apply Gaussian smoothing to LiDAR mask for better coverage
        lidar_evidence = cv2.GaussianBlur(lidar_evidence, (21, 21), 0)
        
        # Normalize after blur
        if lidar_evidence.max() > 0:
            lidar_evidence = lidar_evidence / lidar_evidence.max()
        
        # Weighted fusion
        fused = lidar_weight * lidar_evidence + (1 - lidar_weight) * camera_evidence
        
        # Threshold
        fused_mask = (fused > 0.4).astype(np.uint8) * 255
        
        # Clean up with morphology
        kernel = np.ones((7, 7), np.uint8)
        fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_CLOSE, kernel)
        fused_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_OPEN, kernel)
        
        return fused_mask
    

    """
        Visualize segmentation results
    """
    def visualize_results(self, image: np.ndarray,
                         camera_mask: np.ndarray,
                         lidar_mask: np.ndarray,
                         fused_mask: np.ndarray,
                         pixel_coords: np.ndarray = None,
                         depths: np.ndarray = None):
        _, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Original image with overlayed LiDAR depth points
        if pixel_coords is not None and depths is not None:
            overlay = image.copy()
            depths_norm = (depths - depths.min()) / (depths.max() - depths.min())
            for pt, d in zip(pixel_coords, depths_norm):
                x, y = int(pt[0]), int(pt[1])
                colormap = mpl.colormaps['turbo']
                rgba = colormap(d)
                color = tuple(int(255 * c) for c in rgba[:3])
                cv2.circle(overlay, (x, y), 1, color, -1)
            axes[0, 1].imshow(overlay)
            axes[0, 1].set_title('LiDAR Points (colored by depth)')
        else:
            axes[0, 1].imshow(image)
            axes[0, 1].set_title('Original Image')
        axes[0, 1].axis('off')
        
        # Camera segmentation
        axes[0, 2].imshow(camera_mask, cmap='gray')
        axes[0, 2].set_title('Camera-based Segmentation')
        axes[0, 2].axis('off')
        
        # LiDAR segmentation
        axes[1, 0].imshow(lidar_mask, cmap='gray')
        axes[1, 0].set_title('LiDAR-based Segmentation')
        axes[1, 0].axis('off')
        
        # Fused segmentation mask
        axes[1, 1].imshow(fused_mask, cmap='gray')
        axes[1, 1].set_title('Fused Segmentation')
        axes[1, 1].axis('off')
        
        # Overlay fused mask on original image
        overlay = image.copy()
        overlay[fused_mask > 0] = overlay[fused_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        axes[1, 2].imshow(overlay.astype(np.uint8))
        axes[1, 2].set_title('Fused Result (Green Overlay)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()


"""
    Start point of execution of the pipeline for KITTI dataset
"""
def main():
    # Configurable constant to point at the KITTI dataset
    BASE_PATH = r'./kitti_raw'  # first folder of the dataset
    DATE = '2011_09_26'  # Date folder
    DRIVE = '0005'  # Drive number (with leading zeros)
    CAMERA = 'image_02' # From which camera take the pictures
    FRAME_IDX = 129  # Which frame to process
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
    ground_mask = segmenter.temp_segment_lidar(frame['points'])
    ground_points = np.sum(ground_mask)
    print(f"   Ground points: {ground_points}/{len(ground_mask)} ({100*ground_points/len(ground_mask):.1f}%)")
    
    # Match ground mask with projected points using indices
    print("\n5. Mapping ground mask to projected points...")
    ground_mask_projected = ground_mask[valid_indices]
    print(f"   Projected ground points: {np.sum(ground_mask_projected)}/{len(ground_mask_projected)} ({100*np.sum(ground_mask_projected)/len(ground_mask_projected):.1f}%)")
    
    # Match ground mask with projected points
    # Find which projected points are ground points
    ground_mask_projected = np.zeros(len(valid_points), dtype=bool)
    for i, pt_3d in enumerate(tqdm(valid_points, desc="Creating ground mask", unit="pt")):
        # Find closest point in original cloud
        distances = np.linalg.norm(frame['points'][:, :3] - pt_3d, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 0.1:  # Match tolerance
            ground_mask_projected[i] = ground_mask[closest_idx]
    
    print(f"   Projected ground points: {np.sum(ground_mask_projected)}")

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
    fused_mask = segmenter.fuse_segmentations(
        lidar_mask,
        camera_mask,
        lidar_weight=0.6
    )
    
    # Visualize
    print("\n9. Visualizing results...")
    segmenter.visualize_results(
        frame['image'],
        camera_mask,
        lidar_mask,
        fused_mask,
        pixel_coords,
        depths
    )
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
