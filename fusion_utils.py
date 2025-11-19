"""
Utility functions for KITTI LiDAR-Camera fusion and evaluation
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

"""
    Overlay projected LiDAR points on camera image
    
    Args:
        image: RGB image
        pixel_coords: Projected 2D points (M, 2)
        depths: Depth values for each point
        ground_mask: Optional mask to color ground points differently
        point_size: Size of points to draw
        
    Returns:
        Image with overlaid points
"""
def overlay_lidar_on_image(image: np.ndarray,
                          pixel_coords: np.ndarray,
                          depths: np.ndarray,
                          point_size: int = 2) -> np.ndarray:
    overlay = image.copy()
    
    # Normalize depths for coloring
    if len(depths) > 0:
        depths_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)
    else:
        return overlay
    
    for pt, depth in zip(pixel_coords, depths_norm):
        x, y = int(pt[0]), int(pt[1])
        
        # Color by depth (blue=close, red=far)
        colormap = cm.get_cmap('turbo')
        rgba = colormap(depth)
        color = tuple(int(255 * c) for c in rgba[:3])
        cv2.circle(overlay, (x, y), point_size, color, -1)
    
    return overlay

"""
    Evaluate segmentation performance
    
    Args:
        pred_mask: Predicted binary mask
        
    Returns:
        Dictionary with evaluation metrics
"""
def evaluate_segmentation(pred_mask: np.ndarray) -> dict:
    pred = (pred_mask > 0).astype(bool)
    
    metrics = {
        'coverage': np.sum(pred) / pred.size,
        'num_pixels': np.sum(pred)
    }
    
    return metrics

"""
    Compare different segmentation methods
    
    Returns:
        Dictionary with comparison metrics
"""
def compare_methods(lidar_mask: np.ndarray,
                   camera_mask: np.ndarray,
                   fused_mask: np.ndarray) -> dict:
    metrics = {
        'lidar': evaluate_segmentation(lidar_mask),
        'camera': evaluate_segmentation(camera_mask),
        'fused': evaluate_segmentation(fused_mask)
    }
    
    # Agreement between methods
    lidar_bin = lidar_mask > 0
    camera_bin = camera_mask > 0
    fused_bin = fused_mask > 0
    
    metrics['lidar_camera_agreement'] = np.sum(lidar_bin == camera_bin) / lidar_bin.size
    metrics['fusion_improvement'] = (
        metrics['fused']['coverage'] - 
        max(metrics['lidar']['coverage'], metrics['camera']['coverage'])
    )
    
    return metrics

"""
    Create comprehensive comparison visualization
"""
def visualize_comparison(image: np.ndarray,
                        lidar_mask: np.ndarray,
                        camera_mask: np.ndarray,
                        fused_mask: np.ndarray,
                        pixel_coords: np.ndarray,
                        depths: np.ndarray,
                        save_path: str = None):
    plt.figure(figsize=(20, 12))
    
    # Row 1: Masks
    plt.subplot(3, 4, 1)
    plt.imshow(image)
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(camera_mask, cmap='gray')
    plt.title('Camera Segmentation', fontsize=12)
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(lidar_mask, cmap='gray')
    plt.title('LiDAR Segmentation', fontsize=12)
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(fused_mask, cmap='gray')
    plt.title('Fused Segmentation', fontsize=12)
    plt.axis('off')
    
    # Row 2: Overlays
    overlay = overlay_lidar_on_image(image, pixel_coords, depths)
    plt.subplot(3, 4, 5)
    plt.imshow(overlay.astype(np.uint8))
    plt.title('LiDAR point projection', fontsize=12)
    plt.axis('off')

    camera_overlay = image.copy()
    camera_overlay[camera_mask > 0] = camera_overlay[camera_mask > 0] * 0.5 + np.asarray([255, 0, 0]) * 0.5
    plt.subplot(3, 4, 6)
    plt.imshow(camera_overlay.astype(np.uint8))
    plt.title('Camera Overlay (Red)', fontsize=12)
    plt.axis('off')
    
    lidar_overlay = image.copy()
    lidar_overlay[lidar_mask > 0] = lidar_overlay[lidar_mask > 0] * 0.5 + np.asarray([0, 0, 255]) * 0.5
    plt.subplot(3, 4, 7)
    plt.imshow(lidar_overlay.astype(np.uint8))
    plt.title('LiDAR Overlay (Blue)', fontsize=12)
    plt.axis('off')
    
    fused_overlay = image.copy()
    fused_overlay[fused_mask > 0] = fused_overlay[fused_mask > 0] * 0.5 + np.asarray([0, 255, 0]) * 0.5
    plt.subplot(3, 4, 8)
    plt.imshow(fused_overlay.astype(np.uint8))
    plt.title('Fused Overlay (Green)', fontsize=12)
    plt.axis('off')
    
    # Row 3: Agreement analysis
    agreement_map = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    lidar_bin = lidar_mask > 0
    camera_bin = camera_mask > 0
    
    agreement_map[lidar_bin & camera_bin] = [0, 255, 0]  # Both agree
    agreement_map[lidar_bin & ~camera_bin] = [0, 0, 255]  # LiDAR only
    agreement_map[~lidar_bin & camera_bin] = [255, 0, 0]  # Camera only
    
    plt.subplot(3, 4, 10)
    plt.imshow(agreement_map)
    plt.title('Sensor Agreement\n(Green=Both, Blue=LiDAR, Red=Camera)', fontsize=10)
    plt.axis('off')
    
    # Metrics
    metrics = compare_methods(lidar_mask, camera_mask, fused_mask)
    metrics_text = "Coverage:\n"
    metrics_text += f"  Camera: {metrics['camera']['coverage']*100:.1f}%\n"
    metrics_text += f"  LiDAR:  {metrics['lidar']['coverage']*100:.1f}%\n"
    metrics_text += f"  Fused:  {metrics['fused']['coverage']*100:.1f}%\n"
    metrics_text += f"\nAgreement: {metrics['lidar_camera_agreement']*100:.1f}%"
    
    plt.subplot(3, 4, 11)
    plt.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center')
    plt.axis('off')
    plt.title('Metrics', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()