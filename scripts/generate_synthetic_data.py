"""
Generate synthetic crack data for training and testing

Creates:
- Crack images with realistic appearance
- Segmentation masks
- Metadata (brick type, age, etc.)
- Growth labels (simulated)

Why synthetic data?
- No publicly available labeled crack growth dataset
- Allows testing full pipeline
- Real deployment would use actual before/after measurements
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tqdm import tqdm
import config


def create_synthetic_wall_texture(size=(512, 512)):
    """
    Create realistic wall texture
    """
    h, w = size
    
    # Base color (concrete/brick color)
    base_color = np.random.randint(180, 220)
    wall = np.ones((h, w, 3), dtype=np.uint8) * base_color
    
    # Add Perlin-like noise for texture
    noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
    wall = np.clip(wall.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add some "bricks" pattern (optional)
    if np.random.random() > 0.5:
        for i in range(0, h, 40):
            cv2.line(wall, (0, i), (w, i), (base_color - 20, base_color - 20, base_color - 20), 1)
        for j in range(0, w, 60):
            cv2.line(wall, (j, 0), (j, h), (base_color - 20, base_color - 20, base_color - 20), 1)
    
    return wall


def create_synthetic_crack(size=(512, 512), complexity='medium'):
    """
    Create synthetic crack with realistic morphology
    
    Complexity levels:
    - simple: Single straight line
    - medium: Line with small branches
    - complex: Multiple branches, curved
    """
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Random crack parameters
    crack_width = np.random.randint(*config.SYNTHETIC_CONFIG['crack_width_range'])
    
    if complexity == 'simple':
        # Single line
        x1, y1 = np.random.randint(50, 150), np.random.randint(100, h - 100)
        x2, y2 = np.random.randint(w - 150, w - 50), np.random.randint(100, h - 100)
        cv2.line(mask, (x1, y1), (x2, y2), 1, crack_width)
    
    elif complexity == 'medium':
        # Main crack with 1-2 branches
        x1, y1 = np.random.randint(50, 150), np.random.randint(100, h - 100)
        x2, y2 = np.random.randint(w // 2, w // 2 + 100), np.random.randint(100, h - 100)
        x3, y3 = np.random.randint(w - 150, w - 50), np.random.randint(100, h - 100)
        
        # Draw polyline
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], dtype=np.int32)
        cv2.polylines(mask, [pts], False, 1, crack_width)
        
        # Add small branch
        if np.random.random() > 0.5:
            branch_x = x2 + np.random.randint(-50, 50)
            branch_y = y2 + np.random.randint(-100, 100)
            cv2.line(mask, (x2, y2), (branch_x, branch_y), 1, crack_width - 1)
    
    else:  # complex
        # Create curved crack with multiple segments
        num_points = np.random.randint(4, 7)
        points = []
        for i in range(num_points):
            x = int(50 + (w - 100) * i / (num_points - 1) + np.random.randint(-50, 50))
            y = int(h // 2 + np.random.randint(-150, 150))
            points.append([x, y])
        
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(mask, [pts], False, 1, crack_width)
    
    return mask


def add_crack_to_wall(wall, crack_mask):
    """
    Overlay crack on wall texture
    """
    wall_with_crack = wall.copy()
    
    # Darken crack area
    crack_color_reduction = np.random.randint(40, 80)
    wall_with_crack[crack_mask > 0] = np.clip(
        wall_with_crack[crack_mask > 0].astype(np.int16) - crack_color_reduction,
        0, 255
    ).astype(np.uint8)
    
    # Add some noise to crack area
    crack_area = crack_mask > 0
    noise = np.random.randint(-15, 15, wall_with_crack[crack_area].shape, dtype=np.int16)
    wall_with_crack[crack_area] = np.clip(
        wall_with_crack[crack_area].astype(np.int16) + noise,
        0, 255
    ).astype(np.uint8)
    
    return wall_with_crack


def simulate_crack_growth_synthetic(
    initial_crack_mask,
    brick_type,
    wall_age,
    mortar_type,
    exposure,
    humidity,
    load_bearing
):
    """
    Simulate ground-truth crack growth for training data
    
    Uses engineering heuristics to generate realistic growth amounts
    """
    # Base growth rates (mm per year)
    base_length_growth = np.random.uniform(5, 20)
    base_width_growth = np.random.uniform(0.1, 0.8)
    
    # Apply material factors
    brick_factor = config.BRICK_DURABILITY.get(brick_type, 1.0)
    mortar_factor = config.MORTAR_STRENGTH.get(mortar_type, 1.0)
    exposure_factor = config.EXPOSURE_MULTIPLIER.get(exposure, 1.0)
    humidity_factor = config.HUMIDITY_MULTIPLIER.get(humidity, 1.0)
    
    # Age factor (older walls crack faster)
    age_factor = 1.0 + (wall_age / 50.0)
    
    # Load factor
    load_factor = 1.3 if load_bearing else 1.0
    
    # Combined factors
    total_factor = brick_factor * mortar_factor * exposure_factor * humidity_factor * age_factor * load_factor
    
    # Calculate growth
    length_growth = base_length_growth * total_factor * np.random.uniform(0.8, 1.2)
    width_growth = base_width_growth * total_factor * np.random.uniform(0.8, 1.2)
    
    # Ensure reasonable bounds
    length_growth = np.clip(length_growth, 0, 50)
    width_growth = np.clip(width_growth, 0, 2)
    
    return length_growth, width_growth


def generate_synthetic_dataset(
    num_samples,
    output_dir,
    size=(512, 512)
):
    """
    Generate complete synthetic dataset
    """
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    metadata_list = []
    
    print(f"Generating {num_samples} synthetic samples...")
    
    for i in tqdm(range(num_samples)):
        # Generate random metadata
        brick_type = np.random.choice(config.BRICK_TYPES)
        wall_age = np.random.randint(1, 60)
        mortar_type = np.random.choice(config.MORTAR_TYPES)
        exposure = np.random.choice(config.EXPOSURE_TYPES)
        humidity = np.random.choice(config.HUMIDITY_LEVELS)
        load_bearing = np.random.choice([True, False])
        
        # Create wall and crack
        wall = create_synthetic_wall_texture(size)
        complexity = np.random.choice(['simple', 'medium', 'complex'], p=[0.3, 0.5, 0.2])
        crack_mask = create_synthetic_crack(size, complexity)
        
        # Combine
        image_with_crack = add_crack_to_wall(wall, crack_mask)
        
        # Simulate growth (for training labels)
        length_growth, width_growth = simulate_crack_growth_synthetic(
            crack_mask,
            brick_type,
            wall_age,
            mortar_type,
            exposure,
            humidity,
            load_bearing
        )
        
        # Save image and mask
        image_filename = f'crack_{i:04d}.jpg'
        mask_filename = f'crack_{i:04d}.png'
        
        Image.fromarray(image_with_crack).save(os.path.join(images_dir, image_filename))
        Image.fromarray(crack_mask * 255).save(os.path.join(masks_dir, mask_filename))
        
        # Store metadata
        metadata_list.append({
            'image_id': f'crack_{i:04d}',
            'brick_type': brick_type,
            'wall_age': wall_age,
            'mortar_type': mortar_type,
            'exposure': exposure,
            'humidity': humidity,
            'load_bearing': load_bearing,
            'length_growth_mm': length_growth,
            'width_growth_mm': width_growth
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    print(f"\nDataset generated successfully!")
    print(f"Images: {images_dir}")
    print(f"Masks: {masks_dir}")
    print(f"Metadata: {os.path.join(output_dir, 'metadata.csv')}")
    
    return metadata_df


if __name__ == '__main__':
    # Generate synthetic dataset
    output_dir = config.SYNTHETIC_DATA_DIR
    num_samples = config.SYNTHETIC_CONFIG['num_samples']
    
    metadata = generate_synthetic_dataset(
        num_samples=num_samples,
        output_dir=output_dir,
        size=config.SYNTHETIC_CONFIG['image_size']
    )
    
    print("\nDataset statistics:")
    print(metadata.describe())
    
    print("\nBrick type distribution:")
    print(metadata['brick_type'].value_counts())
    
    print("\nExposure distribution:")
    print(metadata['exposure'].value_counts())
