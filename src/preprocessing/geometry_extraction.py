"""
Crack geometry extraction using OpenCV

Extracts:
- Crack skeleton (centerline)
- Endpoints (propagation points)
- Crack width distribution
- Crack direction vectors
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import label, regionprops
import config


class CrackGeometryExtractor:
    """
    Extract geometric features from crack segmentation mask
    """
    
    def __init__(self, pixel_to_mm=None):
        self.pixel_to_mm = pixel_to_mm or config.PIXEL_TO_MM
        
    def extract_all_features(self, mask):
        """
        Extract all geometric features from crack mask
        
        Args:
            mask: Binary mask (H, W) with values 0 or 1
            
        Returns:
            dict with geometric features
        """
        # Ensure binary mask
        mask = (mask > 0.5).astype(np.uint8)
        
        # Get skeleton
        skeleton = self.get_skeleton(mask)
        
        # Get endpoints
        endpoints = self.find_endpoints(skeleton)
        
        # Get crack width
        width_map, avg_width = self.estimate_width(mask, skeleton)
        
        # Get direction vectors at endpoints
        directions = self.get_endpoint_directions(skeleton, endpoints)
        
        # Calculate crack length
        crack_length_px = np.sum(skeleton > 0)
        crack_length_mm = crack_length_px * self.pixel_to_mm
        
        # Calculate total crack area
        crack_area_px = np.sum(mask > 0)
        crack_area_mm2 = crack_area_px * (self.pixel_to_mm ** 2)
        
        return {
            'skeleton': skeleton,
            'endpoints': endpoints,
            'width_map': width_map,
            'avg_width_px': avg_width,
            'avg_width_mm': avg_width * self.pixel_to_mm,
            'directions': directions,
            'length_px': crack_length_px,
            'length_mm': crack_length_mm,
            'area_px': crack_area_px,
            'area_mm2': crack_area_mm2,
            'num_endpoints': len(endpoints)
        }
    
    def get_skeleton(self, mask):
        """
        Compute morphological skeleton of crack
        
        Why skeletonization?
        - Reduces crack to 1-pixel width centerline
        - Preserves topology (connectivity)
        - Enables endpoint detection
        - Provides crack direction
        """
        skeleton = skeletonize(mask > 0).astype(np.uint8)
        return skeleton
    
    def find_endpoints(self, skeleton):
        """
        Find crack endpoints (tips where crack can propagate)
        
        Method: Endpoint has exactly 1 neighbor in 8-connectivity
        
        Civil engineering insight:
        - Stress concentrates at crack tips
        - Crack growth initiates from endpoints
        - Multiple endpoints = branching cracks
        """
        # Create 3x3 kernel for neighbor counting
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0  # Don't count center pixel
        
        # Count neighbors for each skeleton pixel
        neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
        
        # Endpoint: skeleton pixel with exactly 1 neighbor
        endpoints_mask = (skeleton > 0) & (neighbor_count == 1)
        
        # Get coordinates
        endpoints = np.argwhere(endpoints_mask)
        
        return endpoints  # [(y, x), ...]
    
    def estimate_width(self, mask, skeleton):
        """
        Estimate crack width using distance transform
        
        Method:
        - Distance transform: distance from each pixel to nearest background
        - For skeleton pixels, distance ≈ half of crack width
        - Width at point = 2 * distance_transform_value
        
        Why this method?
        - Fast and robust
        - Handles varying crack widths
        - No manual measurement needed
        """
        # Distance transform: distance to nearest zero pixel
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Width map: for skeleton pixels, width = 2 * distance
        width_map = np.zeros_like(skeleton, dtype=np.float32)
        skeleton_coords = skeleton > 0
        width_map[skeleton_coords] = 2 * distance[skeleton_coords]
        
        # Average width (only where skeleton exists)
        avg_width = np.mean(width_map[skeleton_coords]) if np.any(skeleton_coords) else 0
        
        return width_map, avg_width
    
    def get_endpoint_directions(self, skeleton, endpoints, window_size=10):
        """
        Compute crack propagation direction at each endpoint
        
        Method:
        - Look at nearby skeleton pixels (within window)
        - Fit line to get direction vector
        - Direction points away from existing crack
        
        Why important?
        - Crack grows in direction of principal stress
        - Usually continues in same direction
        - Small deviation expected (±15 degrees)
        """
        directions = []
        
        for ep in endpoints:
            y, x = ep
            
            # Get nearby skeleton pixels
            y_min = max(0, y - window_size)
            y_max = min(skeleton.shape[0], y + window_size + 1)
            x_min = max(0, x - window_size)
            x_max = min(skeleton.shape[1], x + window_size + 1)
            
            window = skeleton[y_min:y_max, x_min:x_max]
            coords = np.argwhere(window > 0)
            
            if len(coords) < 2:
                # Not enough points, use default direction
                directions.append([0, 1])  # Horizontal
                continue
            
            # Convert to global coordinates
            coords[:, 0] += y_min
            coords[:, 1] += x_min
            
            # Remove endpoint itself
            coords = coords[~np.all(coords == ep, axis=1)]
            
            if len(coords) < 1:
                directions.append([0, 1])
                continue
            
            # Compute direction vector (from centroid to endpoint)
            centroid = np.mean(coords, axis=0)
            direction = ep - centroid
            
            # Normalize
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction = np.array([0, 1])
            
            directions.append(direction)
        
        return np.array(directions)
    
    def get_morphological_features(self, mask):
        """
        Extract additional morphological features for ML model
        
        Features useful for growth prediction:
        - Perimeter-to-area ratio (tortuosity)
        - Eccentricity (straight vs curved)
        - Orientation (angle)
        """
        # Label connected components
        labeled = label(mask)
        
        if labeled.max() == 0:
            return {
                'perimeter': 0,
                'area': 0,
                'perimeter_area_ratio': 0,
                'eccentricity': 0,
                'orientation': 0,
                'solidity': 0
            }
        
        # Get properties of largest region (main crack)
        regions = regionprops(labeled)
        main_region = max(regions, key=lambda r: r.area)
        
        perimeter = main_region.perimeter
        area = main_region.area
        
        return {
            'perimeter': perimeter,
            'area': area,
            'perimeter_area_ratio': perimeter / (area + 1e-6),
            'eccentricity': main_region.eccentricity,
            'orientation': main_region.orientation,
            'solidity': main_region.solidity
        }


def visualize_geometry(image, geometry_features):
    """
    Visualize extracted geometry on image
    """
    vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw skeleton in blue
    skeleton_mask = geometry_features['skeleton'] > 0
    vis[skeleton_mask] = [255, 0, 0]  # Blue
    
    # Draw endpoints in red
    for ep in geometry_features['endpoints']:
        cv2.circle(vis, (ep[1], ep[0]), 5, (0, 0, 255), -1)
    
    # Draw direction vectors in green
    for ep, direction in zip(geometry_features['endpoints'], geometry_features['directions']):
        end_point = (ep + direction * 30).astype(int)
        cv2.arrowedLine(vis, (ep[1], ep[0]), (end_point[1], end_point[0]), (0, 255, 0), 2)
    
    return vis


if __name__ == '__main__':
    # Test on synthetic crack
    import matplotlib.pyplot as plt
    
    # Create simple crack mask
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.line(mask, (100, 100), (400, 300), 1, 5)
    
    # Extract features
    extractor = CrackGeometryExtractor()
    features = extractor.extract_all_features(mask)
    
    print("Crack Features:")
    print(f"Length: {features['length_mm']:.2f} mm")
    print(f"Average width: {features['avg_width_mm']:.2f} mm")
    print(f"Number of endpoints: {features['num_endpoints']}")
    print(f"Endpoints: {features['endpoints']}")
    
    # Visualize
    vis = visualize_geometry(mask * 255, features)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('Original Mask')
    plt.subplot(1, 2, 2)
    plt.imshow(vis)
    plt.title('Extracted Geometry')
    plt.tight_layout()
    plt.savefig('/home/claude/test_geometry.png')
    print("Saved visualization to test_geometry.png")
