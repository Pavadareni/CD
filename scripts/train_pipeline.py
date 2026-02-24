"""
Complete training pipeline for crack growth prediction system

Steps:
1. Train U-Net segmentation model
2. Extract features from all images
3. Train XGBoost growth prediction model
4. Save all models
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
from PIL import Image
import config
from models.segmentation.train_segmentation import train_segmentation_model
from models.prediction.growth_model import CrackGrowthPredictor
from src.preprocessing.geometry_extraction import CrackGeometryExtractor
from src.preprocessing.feature_engineering import FeatureEngineer


def train_segmentation(data_dir):
    """
    Step 1: Train U-Net segmentation model
    """
    print("="*80)
    print("STEP 1: TRAINING CRACK SEGMENTATION MODEL (U-Net)")
    print("="*80)
    
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    model = train_segmentation_model(
        train_image_dir=images_dir,
        train_mask_dir=masks_dir,
        num_epochs=config.UNET_CONFIG['num_epochs'],
        batch_size=config.UNET_CONFIG['batch_size'],
        learning_rate=config.UNET_CONFIG['learning_rate']
    )
    
    print("\n✓ Segmentation model training complete!")
    return model


def extract_features_from_dataset(data_dir, segmentation_model=None):
    """
    Step 2: Extract geometric features from all images
    """
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING FEATURES FROM DATASET")
    print("="*80)
    
    # Load metadata
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    metadata = pd.read_csv(metadata_path)
    
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    # Initialize extractors
    geom_extractor = CrackGeometryExtractor()
    feat_engineer = FeatureEngineer()
    
    features_list = []
    labels_list = []
    
    print(f"Processing {len(metadata)} images...")
    
    for idx, row in metadata.iterrows():
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(metadata)} images...")
        
        # Load mask
        mask_path = os.path.join(masks_dir, f"{row['image_id']}.png")
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.uint8)
        
        # Extract geometry
        geom_features = geom_extractor.extract_all_features(mask)
        geom_features['morphological'] = geom_extractor.get_morphological_features(mask)
        
        # Engineer features
        features = feat_engineer.create_features(
            geometry_features=geom_features,
            brick_type=row['brick_type'],
            wall_age=row['wall_age'],
            mortar_type=row['mortar_type'],
            exposure=row['exposure'],
            humidity=row['humidity'],
            load_bearing=row['load_bearing']
        )
        
        features_list.append(features)
        
        # Labels
        labels_list.append({
            'length_growth_mm': row['length_growth_mm'],
            'width_growth_mm': row['width_growth_mm']
        })
    
    # Convert to DataFrames
    features_df = feat_engineer.features_to_dataframe(features_list)
    labels_df = pd.DataFrame(labels_list)
    
    # Save processed features
    processed_dir = config.PROCESSED_DATA_DIR
    os.makedirs(processed_dir, exist_ok=True)
    
    features_df.to_csv(os.path.join(processed_dir, 'features.csv'), index=False)
    labels_df.to_csv(os.path.join(processed_dir, 'labels.csv'), index=False)
    
    print(f"\n✓ Feature extraction complete!")
    print(f"  Features shape: {features_df.shape}")
    print(f"  Saved to: {processed_dir}")
    
    return features_df, labels_df


def train_prediction_model(features_df, labels_df):
    """
    Step 3: Train XGBoost growth prediction model
    """
    print("\n" + "="*80)
    print("STEP 3: TRAINING GROWTH PREDICTION MODEL (XGBoost)")
    print("="*80)
    
    # Prepare data
    X = features_df.values
    y_length = labels_df['length_growth_mm'].values
    y_width = labels_df['width_growth_mm'].values
    
    # Calculate risk categories
    total_growth = y_length + y_width * 10
    y_risk = np.zeros_like(total_growth, dtype=int)
    y_risk[total_growth >= config.RISK_THRESHOLDS['low']] = 1
    y_risk[total_growth >= config.RISK_THRESHOLDS['medium']] = 2
    
    # Train model
    predictor = CrackGrowthPredictor()
    predictor.train(X, y_length, y_width, y_risk)
    
    # Save model
    model_path = os.path.join(config.MODELS_DIR, 'growth_predictor.pkl')
    predictor.save(model_path)
    
    # Show feature importance
    print("\n" + "-"*80)
    print("TOP 15 MOST IMPORTANT FEATURES:")
    print("-"*80)
    importance = predictor.get_feature_importance(top_n=15)
    for i, (feature, score) in enumerate(importance, 1):
        feature_name = features_df.columns[int(feature.replace('f', ''))]
        print(f"{i:2d}. {feature_name:30s}: {score:6.0f}")
    
    print(f"\n✓ Prediction model training complete!")
    print(f"  Model saved to: {model_path}")
    
    return predictor


def main():
    """
    Run complete training pipeline
    """
    print("\n" + "="*80)
    print("CRACK GROWTH PREDICTION SYSTEM - TRAINING PIPELINE")
    print("="*80)
    
    # Check if synthetic data exists
    data_dir = config.SYNTHETIC_DATA_DIR
    if not os.path.exists(os.path.join(data_dir, 'metadata.csv')):
        print("\nERROR: Synthetic data not found!")
        print("Please run: python scripts/generate_synthetic_data.py")
        return
    
    # Step 1: Train segmentation
    try:
        segmentation_model = train_segmentation(data_dir)
    except Exception as e:
        print(f"\n⚠ Warning: Segmentation training failed: {e}")
        print("Continuing with feature extraction using ground truth masks...")
        segmentation_model = None
    
    # Step 2: Extract features
    features_df, labels_df = extract_features_from_dataset(data_dir, segmentation_model)
    
    # Step 3: Train prediction model
    predictor = train_prediction_model(features_df, labels_df)
    
    print("\n" + "="*80)
    print("✓ TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print("\nTrained models:")
    print(f"  1. Segmentation model: {config.MODELS_DIR}/unet_best.pth")
    print(f"  2. Growth predictor:   {config.MODELS_DIR}/growth_predictor.pkl")
    print("\nYou can now run predictions using:")
    print("  python scripts/predict.py --image <path_to_image> --brick_type clay ...")
    print("="*80)


if __name__ == '__main__':
    main()
