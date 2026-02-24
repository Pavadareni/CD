"""
Crack growth simulation - geometry-based propagation

Simulates crack extension and widening based on:
1. ML predictions (length and width growth)
2. Physical constraints (direction, endpoints)
3. Civil engineering principles

NOT using GANs or diffusion - purely geometric simulation
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import cv2
import config


class CrackGrowthSimulator:
    """
    Simulate crack growth based on predictions
    """
    
    def __init__(self, angle_variance=None, growth_steps=None):
        self.angle_variance = angle_variance or config.GROWTH_CONFIG['extension_angle_variance']
        self.growth_steps = growth_steps or config.GROWTH_CONFIG['growth_steps']
        
    def simulate_growth(
        self,
        original_mask,
        skeleton,
        endpoints,
        directions,
        predicted_length_growth_px,
        predicted_width_growth_px
    ):
        """
        Simulate crack growth based on predictions
        
        Args:
            original_mask: Original crack binary mask (H, W)
            skeleton: Crack skeleton (H, W)
            endpoints: Array of endpoint coordinates [(y, x), ...]
            directions: Direction vectors at endpoints [(dy, dx), ...]
            predicted_length_growth_px: Predicted length growth in pixels
            predicted_width_growth_px: Predicted width growth in pixels
            
        Returns:
            growth_mask: Binary mask showing ONLY the new growth (not original crack)
        """
        h, w = original_mask.shape
        growth_mask = np.zeros_like(original_mask, dtype=np.uint8)
        
        # If no endpoints, cannot simulate growth
        if len(endpoints) == 0:
            return growth_mask
        
        # =====================================================
        # 1. CRACK EXTENSION (length growth)
        # =====================================================
        # Civil engineering principle:
        # - Cracks propagate from stress concentration points (endpoints)
        # - Growth direction follows existing crack direction
        # - Small angular deviation allowed (material heterogeneity)
        
        # Distribute growth among endpoints
        growth_per_endpoint = predicted_length_growth_px / len(endpoints)
        
        for endpoint, direction in zip(endpoints, directions):
            self._extend_crack_from_endpoint(
                growth_mask,
                endpoint,
                direction,
                growth_per_endpoint,
                h, w
            )
        
        # =====================================================
        # 2. CRACK WIDENING (width growth)
        # =====================================================
        # Civil engineering principle:
        # - Existing crack widens due to continued stress
        # - Widening is relatively uniform along crack length
        # - Use morphological dilation
        
        if predicted_width_growth_px > 0:
            widened_mask = self._widen_crack(
                original_mask,
                predicted_width_growth_px
            )
            
            # Add widening to growth (excluding original crack)
            growth_mask = np.maximum(growth_mask, widened_mask)
            growth_mask[original_mask > 0] = 0  # Remove overlap with original
        
        return growth_mask
    
    def _extend_crack_from_endpoint(
        self,
        growth_mask,
        endpoint,
        direction,
        growth_length_px,
        height,
        width
    ):
        """
        Extend crack from a single endpoint
        
        Method:
        - Start at endpoint
        - Move in direction vector
        - Add small random deviation (realistic crack path)
        - Draw line segments
        """
        y_start, x_start = endpoint
        dy, dx = direction
        
        # Current position
        y_curr, x_curr = float(y_start), float(x_start)
        
        # Step size for smooth growth
        step_size = max(2, growth_length_px / self.growth_steps)
        num_steps = int(growth_length_px / step_size)
        
        for step in range(num_steps):
            # Small angular deviation (simulate material heterogeneity)
            angle_deviation = np.random.uniform(
                -self.angle_variance,
                self.angle_variance
            )
            angle_rad = np.radians(angle_deviation)
            
            # Rotate direction vector
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            dx_new = dx * cos_a - dy * sin_a
            dy_new = dx * sin_a + dy * cos_a
            
            # Update direction (smooth transition)
            dx = 0.8 * dx + 0.2 * dx_new
            dy = 0.8 * dy + 0.2 * dy_new
            
            # Normalize
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                dx /= norm
                dy /= norm
            
            # Move in direction
            y_next = y_curr + dy * step_size
            x_next = x_curr + dx * step_size
            
            # Check bounds
            if not (0 <= y_next < height and 0 <= x_next < width):
                break
            
            # Draw line segment
            pt1 = (int(x_curr), int(y_curr))
            pt2 = (int(x_next), int(y_next))
            cv2.line(growth_mask, pt1, pt2, 1, thickness=2)
            
            # Update position
            y_curr, x_curr = y_next, x_next
    
    def _widen_crack(self, original_mask, width_growth_px):
        """
        Widen existing crack using morphological dilation
        
        Method:
        - Create circular structuring element
        - Dilate crack by predicted width growth
        - Subtract original to get only the widening
        """
        # Create structuring element (circle)
        kernel_size = int(np.ceil(width_growth_px))
        if kernel_size < 1:
            return np.zeros_like(original_mask)
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size * 2 + 1, kernel_size * 2 + 1)
        )
        
        # Dilate original crack
        widened = cv2.dilate(original_mask.astype(np.uint8), kernel, iterations=1)
        
        # Get only the new width (not original crack)
        width_growth_mask = np.maximum(widened - original_mask, 0).astype(np.uint8)
        
        return width_growth_mask
    
    def create_smooth_growth(self, growth_mask):
        """
        Apply smoothing to make growth look more natural
        
        Optional post-processing for visual quality
        """
        # Slight Gaussian blur to smooth jagged edges
        smoothed = cv2.GaussianBlur(growth_mask.astype(np.float32), (3, 3), 0.5)
        smoothed = (smoothed > 0.3).astype(np.uint8)
        
        return smoothed


def mm_to_pixels(mm, pixel_to_mm=None):
    """
    Convert millimeters to pixels
    """
    pixel_to_mm = pixel_to_mm or config.PIXEL_TO_MM
    return mm / pixel_to_mm


def pixels_to_mm(pixels, pixel_to_mm=None):
    """
    Convert pixels to millimeters
    """
    pixel_to_mm = pixel_to_mm or config.PIXEL_TO_MM
    return pixels * pixel_to_mm


if __name__ == '__main__':
    # Test crack growth simulation
    from src.preprocessing.geometry_extraction import CrackGeometryExtractor
    import matplotlib.pyplot as plt
    
    # Create test crack
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.line(mask, (100, 256), (300, 256), 1, 4)
    
    # Extract geometry
    extractor = CrackGeometryExtractor()
    geom = extractor.extract_all_features(mask)
    
    # Simulate growth
    simulator = CrackGrowthSimulator()
    
    # Predict 20mm length growth, 0.5mm width growth
    length_growth_px = mm_to_pixels(20)
    width_growth_px = mm_to_pixels(0.5)
    
    growth_mask = simulator.simulate_growth(
        original_mask=mask,
        skeleton=geom['skeleton'],
        endpoints=geom['endpoints'],
        directions=geom['directions'],
        predicted_length_growth_px=length_growth_px,
        predicted_width_growth_px=width_growth_px
    )
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Original Crack')
    
    axes[1].imshow(growth_mask, cmap='Reds')
    axes[1].set_title('Predicted Growth')
    
    combined = np.zeros((512, 512, 3), dtype=np.uint8)
    combined[mask > 0] = [0, 255, 0]  # Green
    combined[growth_mask > 0] = [255, 0, 0]  # Red
    axes[2].imshow(combined)
    axes[2].set_title('Combined View')
    
    plt.tight_layout()
    plt.savefig('/home/claude/test_simulation.png')
    print("Saved simulation test to test_simulation.png")
    print(f"Simulated length growth: {pixels_to_mm(length_growth_px):.2f} mm")
    print(f"Simulated width growth: {pixels_to_mm(width_growth_px):.2f} mm")
