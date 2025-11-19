import os
import cv2
import glob
import numpy as np
from typing import Dict

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
        self.camera = camera.split("_")[1]
        
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
            self.drive_path, f"image_{self.camera}", "data", "*.png")))
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
        if f"P_rect_{self.camera}" in calib:
            calib[f"P{self.camera[1]}"] = calib[f"P_rect_{self.camera}"].reshape(3, 4)
        
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