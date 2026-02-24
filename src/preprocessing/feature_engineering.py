"""
Feature engineering for crack growth prediction

Combines:
- Geometric features from image
- Structural metadata (brick type, mortar, etc.)
- Environmental factors

Engineering principles:
- Material properties affect crack propagation rate
- Environmental exposure accelerates degradation
- Structural load creates stress concentrations
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import config


class FeatureEngineer:
    """
    Engineer features for crack growth prediction model
    """
    
    def __init__(self):
        self.feature_names = []
        
    def create_features(
        self,
        geometry_features,
        brick_type,
        wall_age,
        mortar_type,
        exposure,
        humidity,
        load_bearing
    ):
        """
        Create feature vector for ML model
        
        Args:
            geometry_features: dict from CrackGeometryExtractor
            brick_type: str, one of BRICK_TYPES
            wall_age: int, years
            mortar_type: str, one of MORTAR_TYPES
            exposure: str, 'indoor' or 'outdoor'
            humidity: str, 'low', 'medium', or 'high'
            load_bearing: bool
            
        Returns:
            dict of features
        """
        features = {}
        
        # ==========================================
        # 1. GEOMETRIC FEATURES (from image)
        # ==========================================
        features['crack_length_mm'] = geometry_features['length_mm']
        features['crack_width_mm'] = geometry_features['avg_width_mm']
        features['crack_area_mm2'] = geometry_features['area_mm2']
        features['num_endpoints'] = geometry_features['num_endpoints']
        
        # Morphological features
        morph = geometry_features.get('morphological', {})
        features['perimeter_area_ratio'] = morph.get('perimeter_area_ratio', 0)
        features['eccentricity'] = morph.get('eccentricity', 0)
        features['solidity'] = morph.get('solidity', 0)
        
        # ==========================================
        # 2. MATERIAL PROPERTIES
        # ==========================================
        # Brick durability factor
        features['brick_durability'] = config.BRICK_DURABILITY.get(brick_type, 1.0)
        
        # Mortar strength factor
        features['mortar_strength'] = config.MORTAR_STRENGTH.get(mortar_type, 1.0)
        
        # One-hot encode brick type
        for bt in config.BRICK_TYPES:
            features[f'brick_{bt}'] = 1 if brick_type == bt else 0
        
        # One-hot encode mortar type
        for mt in config.MORTAR_TYPES:
            features[f'mortar_{mt}'] = 1 if mortar_type == mt else 0
        
        # ==========================================
        # 3. AGE AND DEGRADATION
        # ==========================================
        features['wall_age_years'] = wall_age
        features['wall_age_squared'] = wall_age ** 2  # Non-linear aging effect
        
        # Age categories
        features['age_new'] = 1 if wall_age < 5 else 0
        features['age_medium'] = 1 if 5 <= wall_age < 20 else 0
        features['age_old'] = 1 if wall_age >= 20 else 0
        
        # ==========================================
        # 4. ENVIRONMENTAL FACTORS
        # ==========================================
        features['exposure_multiplier'] = config.EXPOSURE_MULTIPLIER.get(exposure, 1.0)
        features['humidity_multiplier'] = config.HUMIDITY_MULTIPLIER.get(humidity, 1.0)
        
        # Exposure flags
        features['is_outdoor'] = 1 if exposure == 'outdoor' else 0
        
        # Humidity level encoding
        features['humidity_low'] = 1 if humidity == 'low' else 0
        features['humidity_medium'] = 1 if humidity == 'medium' else 0
        features['humidity_high'] = 1 if humidity == 'high' else 0
        
        # ==========================================
        # 5. STRUCTURAL LOAD
        # ==========================================
        features['is_load_bearing'] = 1 if load_bearing else 0
        
        # ==========================================
        # 6. INTERACTION FEATURES
        # ==========================================
        # Civil engineering insight: Combine factors that interact
        
        # Outdoor + high humidity = severe exposure (freeze-thaw cycles)
        features['severe_exposure'] = features['is_outdoor'] * features['humidity_high']
        
        # Old age + load bearing = structural fatigue
        features['structural_fatigue'] = (1 if wall_age > 20 else 0) * features['is_load_bearing']
        
        # Weak mortar + environmental exposure
        features['mortar_environmental_risk'] = features['mortar_strength'] * features['exposure_multiplier']
        
        # Combined degradation factor
        features['combined_degradation'] = (
            features['brick_durability'] *
            features['mortar_strength'] *
            features['exposure_multiplier'] *
            features['humidity_multiplier']
        )
        
        # ==========================================
        # 7. CRACK SEVERITY INDICATORS
        # ==========================================
        # Width-to-length ratio
        features['width_length_ratio'] = (
            features['crack_width_mm'] / (features['crack_length_mm'] + 1e-6)
        )
        
        # Severity score (heuristic)
        features['severity_score'] = (
            features['crack_length_mm'] * 0.5 +
            features['crack_width_mm'] * 10 +
            features['num_endpoints'] * 5
        )
        
        return features
    
    def features_to_dataframe(self, features_list):
        """
        Convert list of feature dicts to pandas DataFrame
        """
        df = pd.DataFrame(features_list)
        self.feature_names = df.columns.tolist()
        return df
    
    def get_feature_names(self):
        """
        Get ordered list of feature names
        """
        return self.feature_names


def create_growth_labels(
    initial_length_mm,
    initial_width_mm,
    final_length_mm,
    final_width_mm
):
    """
    Create training labels for supervised learning
    
    For training data, we need before/after measurements
    
    Args:
        initial_length_mm: Initial crack length
        initial_width_mm: Initial crack width
        final_length_mm: Length after 1 year
        final_width_mm: Width after 1 year
        
    Returns:
        dict with growth labels
    """
    length_growth = final_length_mm - initial_length_mm
    width_growth = final_width_mm - initial_width_mm
    
    # Risk category based on total growth
    total_growth = length_growth + width_growth * 10  # Weight width more
    
    if total_growth < config.RISK_THRESHOLDS['low']:
        risk = 0  # Low
    elif total_growth < config.RISK_THRESHOLDS['medium']:
        risk = 1  # Medium
    else:
        risk = 2  # High
    
    return {
        'length_growth_mm': max(0, length_growth),  # Ensure non-negative
        'width_growth_mm': max(0, width_growth),
        'total_growth_mm': max(0, total_growth),
        'risk_category': risk
    }


if __name__ == '__main__':
    # Test feature engineering
    from src.preprocessing.geometry_extraction import CrackGeometryExtractor
    import cv2
    
    # Create test crack
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.line(mask, (100, 100), (400, 300), 1, 5)
    
    # Extract geometry
    extractor = CrackGeometryExtractor()
    geom = extractor.extract_all_features(mask)
    geom['morphological'] = extractor.get_morphological_features(mask)
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(
        geometry_features=geom,
        brick_type='clay',
        wall_age=15,
        mortar_type='cement',
        exposure='outdoor',
        humidity='high',
        load_bearing=True
    )
    
    print("Engineered Features:")
    for name, value in features.items():
        print(f"{name:30s}: {value:.4f}")
    
    print(f"\nTotal features: {len(features)}")
