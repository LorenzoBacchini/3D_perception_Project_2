import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from U_Net.U_Net import UNet
from typing import Tuple, Dict
from torchvision import transforms
from kitti_dataloader import KITTIDataLoader

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
        P2 = calib[f"P{self.loader.camera[1]}"]  # (3, 4)
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

        return torch.Tensor.numpy(road_mask) * 255
    

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
            cv2.circle(mask, (x, y), 3, 255, -1)
        return mask