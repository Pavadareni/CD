"""
Configuration file for crack growth prediction system
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, 'synthetic')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved')

# Image settings
IMAGE_SIZE = (512, 512)  # Resize all images to this size
PIXEL_TO_MM = 0.5  # Conversion factor: 1 pixel = 0.5 mm (calibrated with reference object)

# Segmentation model (U-Net)
UNET_CONFIG = {
    'in_channels': 3,
    'out_channels': 1,  # Binary segmentation
    'init_features': 32,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'num_epochs': 50,
    'val_split': 0.2
}

# Growth prediction model (XGBoost)
XGBOOST_CONFIG = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'reg:squarederror',
    'random_state': 42
}

# Feature categories
BRICK_TYPES = ['clay', 'concrete', 'fly_ash', 'sand_lime', 'engineering_brick']
MORTAR_TYPES = ['cement', 'lime', 'cement_lime_mix']
EXPOSURE_TYPES = ['indoor', 'outdoor']
HUMIDITY_LEVELS = ['low', 'medium', 'high']

# Brick durability factors (engineering coefficients)
# Higher value = more prone to cracking
BRICK_DURABILITY = {
    'clay': 1.2,
    'concrete': 1.0,
    'fly_ash': 0.8,
    'sand_lime': 1.1,
    'engineering_brick': 0.7
}

# Mortar strength factors
MORTAR_STRENGTH = {
    'cement': 1.0,
    'lime': 1.5,  # Weaker, more crack growth
    'cement_lime_mix': 1.2
}

# Environmental multipliers
EXPOSURE_MULTIPLIER = {
    'indoor': 1.0,
    'outdoor': 1.5  # Freeze-thaw, moisture cycles
}

HUMIDITY_MULTIPLIER = {
    'low': 1.0,
    'medium': 1.3,
    'high': 1.6  # Moisture accelerates crack growth
}

# Crack growth simulation
GROWTH_CONFIG = {
    'extension_angle_variance': 15,  # degrees, crack can deviate from original direction
    'min_segment_length': 5,  # pixels
    'growth_steps': 20,  # Number of intermediate steps for smooth visualization
    'width_expansion_uniform': True  # Expand width uniformly vs concentrated at tip
}

# Visualization
VIZ_CONFIG = {
    'existing_crack_color': (0, 255, 0),  # Green for existing
    'predicted_crack_color': (255, 0, 0),  # Red for predicted growth
    'overlay_alpha': 0.6,  # Transparency
    'line_thickness': 2
}

# Risk thresholds (based on crack growth in mm)
RISK_THRESHOLDS = {
    'low': 5,      # < 5mm growth
    'medium': 15,  # 5-15mm growth
    'high': 15     # > 15mm growth
}

# Synthetic data generation
SYNTHETIC_CONFIG = {
    'num_samples': 500,
    'image_size': IMAGE_SIZE,
    'crack_width_range': (1, 5),  # pixels
    'crack_length_range': (50, 300),  # pixels
    'noise_level': 0.1
}
