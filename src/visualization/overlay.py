"""
Visualization module - overlay predicted crack growth on original image

Creates output image that:
- Preserves original image as background
- Shows existing crack (original)
- Overlays predicted growth in different color (red)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import cv2
from PIL import Image
import config


class CrackVisualizer:
    """
    Visualize crack growth predictions on original images
    """
    
    def __init__(
        self,
        existing_color=None,
        predicted_color=None,
        overlay_alpha=None
    ):
        self.existing_color = existing_color or config.VIZ_CONFIG['existing_crack_color']
        self.predicted_color = predicted_color or config.VIZ_CONFIG['predicted_crack_color']
        self.overlay_alpha = overlay_alpha or config.VIZ_CONFIG['overlay_alpha']
        
    def create_overlay(
        self,
        original_image,
        existing_crack_mask,
        predicted_growth_mask,
        predictions=None
    ):
        """
        Create visualization overlay
        
        Args:
            original_image: Original RGB image (H, W, 3) or path to image
            existing_crack_mask: Binary mask of existing crack (H, W)
            predicted_growth_mask: Binary mask of predicted growth (H, W)
            predictions: Optional dict with prediction metrics
            
        Returns:
            output_image: RGB image with overlays (H, W, 3)
        """
        # Load image if path provided
        if isinstance(original_image, str):
            original_image = np.array(Image.open(original_image).convert('RGB'))
        
        # Ensure image is RGB
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Create output image (copy of original)
        output = original_image.copy()
        
        # Create overlay for existing crack (green)
        existing_overlay = np.zeros_like(output)
        existing_overlay[existing_crack_mask > 0] = self.existing_color
        
        # Create overlay for predicted growth (red)
        predicted_overlay = np.zeros_like(output)
        predicted_overlay[predicted_growth_mask > 0] = self.predicted_color
        
        # Blend overlays with original image
        # Existing crack with transparency
        mask_existing = (existing_crack_mask > 0).astype(np.float32)
        output = output * (1 - mask_existing[:, :, np.newaxis] * self.overlay_alpha) + \
                 existing_overlay * mask_existing[:, :, np.newaxis] * self.overlay_alpha
        
        # Predicted growth with transparency (on top)
        mask_predicted = (predicted_growth_mask > 0).astype(np.float32)
        output = output * (1 - mask_predicted[:, :, np.newaxis] * self.overlay_alpha) + \
                 predicted_overlay * mask_predicted[:, :, np.newaxis] * self.overlay_alpha
        
        output = output.astype(np.uint8)
        
        # Add text annotations if predictions provided
        if predictions is not None:
            output = self._add_annotations(output, predictions)
        
        return output
    
    def _add_annotations(self, image, predictions):
        """
        Add text annotations with prediction results
        """
        h, w = image.shape[:2]
        
        # Create semi-transparent panel for text
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = [0, 0, 0]  # Black background
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)  # White text
        
        texts = [
            f"Predicted Length Growth: {predictions.get('length_growth_mm', 0):.2f} mm",
            f"Predicted Width Growth: {predictions.get('width_growth_mm', 0):.3f} mm",
            f"Risk Category: {predictions.get('risk_label', 'Unknown')}",
            f"Confidence: {predictions.get('confidence', 0):.2%}"
        ]
        
        y_offset = 25
        for i, text in enumerate(texts):
            cv2.putText(
                panel,
                text,
                (10, y_offset + i * 28),
                font,
                font_scale,
                color,
                thickness
            )
        
        # Add legend
        legend_y = 25
        cv2.rectangle(panel, (w - 220, legend_y), (w - 200, legend_y + 15), self.existing_color, -1)
        cv2.putText(panel, "Existing Crack", (w - 190, legend_y + 12), font, 0.5, color, 1)
        
        cv2.rectangle(panel, (w - 220, legend_y + 25), (w - 200, legend_y + 40), self.predicted_color, -1)
        cv2.putText(panel, "Predicted Growth", (w - 190, legend_y + 37), font, 0.5, color, 1)
        
        # Combine panel with image
        output = np.vstack([image, panel])
        
        return output
    
    def create_side_by_side(
        self,
        original_image,
        existing_crack_mask,
        predicted_growth_mask,
        predictions=None
    ):
        """
        Create side-by-side comparison: Original | Prediction
        """
        # Create overlay
        overlay = self.create_overlay(
            original_image,
            existing_crack_mask,
            predicted_growth_mask,
            predictions=None  # Don't add annotations to overlay in side-by-side
        )
        
        # Load/prepare original
        if isinstance(original_image, str):
            original = np.array(Image.open(original_image).convert('RGB'))
        else:
            original = original_image.copy()
        
        # Ensure same size
        if original.shape != overlay.shape:
            overlay = cv2.resize(overlay, (original.shape[1], original.shape[0]))
        
        # Stack horizontally
        side_by_side = np.hstack([original, overlay])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side, "Original", (20, 40), font, 1.2, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Predicted Growth (1 Year)", 
                   (original.shape[1] + 20, 40), font, 1.2, (255, 255, 255), 2)
        
        # Add predictions panel at bottom
        if predictions is not None:
            side_by_side = self._add_annotations(side_by_side, predictions)
        
        return side_by_side
    
    def save_visualization(
        self,
        output_image,
        save_path
    ):
        """
        Save visualization to file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        output_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, output_bgr)
        
        print(f"Visualization saved to {save_path}")


if __name__ == '__main__':
    # Test visualization
    import matplotlib.pyplot as plt
    
    # Create test data
    h, w = 512, 512
    
    # Original image (simulated wall texture)
    original = np.random.randint(200, 230, (h, w, 3), dtype=np.uint8)
    
    # Add some texture
    noise = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
    original = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Existing crack
    existing_crack = np.zeros((h, w), dtype=np.uint8)
    cv2.line(existing_crack, (100, 256), (300, 256), 1, 4)
    
    # Predicted growth
    predicted_growth = np.zeros((h, w), dtype=np.uint8)
    cv2.line(predicted_growth, (300, 256), (400, 256), 1, 4)
    cv2.line(predicted_growth, (100, 250), (300, 250), 1, 1)  # Widening
    cv2.line(predicted_growth, (100, 262), (300, 262), 1, 1)
    
    # Create visualizer
    viz = CrackVisualizer()
    
    # Test predictions
    predictions = {
        'length_growth_mm': 12.5,
        'width_growth_mm': 0.35,
        'risk_label': 'High',
        'confidence': 0.87
    }
    
    # Create overlay
    overlay = viz.create_overlay(original, existing_crack, predicted_growth, predictions)
    
    # Create side-by-side
    side_by_side = viz.create_side_by_side(original, existing_crack, predicted_growth, predictions)
    
    # Save
    viz.save_visualization(overlay, '/home/claude/test_overlay.png')
    viz.save_visualization(side_by_side, '/home/claude/test_side_by_side.png')
    
    print("Visualizations saved!")
    print("- test_overlay.png: Single overlay image")
    print("- test_side_by_side.png: Side-by-side comparison")
