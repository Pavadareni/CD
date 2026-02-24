"""
Demo script - Complete example of using the crack growth prediction system

This script demonstrates:
1. Creating a test crack image
2. Running the complete pipeline
3. Visualizing results
4. Interpreting predictions
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import config
from scripts.generate_synthetic_data import create_synthetic_wall_texture, create_synthetic_crack, add_crack_to_wall


def create_demo_crack_image(save_path='demo_crack.jpg'):
    """
    Create a realistic demo crack image for testing
    """
    print("Creating demo crack image...")
    
    # Create wall texture
    wall = create_synthetic_wall_texture(size=(512, 512))
    
    # Create realistic crack
    crack_mask = create_synthetic_crack(size=(512, 512), complexity='medium')
    
    # Add crack to wall
    image_with_crack = add_crack_to_wall(wall, crack_mask)
    
    # Save
    Image.fromarray(image_with_crack).save(save_path)
    print(f"✓ Demo crack image saved to: {save_path}")
    
    return save_path, crack_mask


def demo_with_trained_models():
    """
    Demo using trained models (requires running train_pipeline.py first)
    """
    from scripts.predict import CrackGrowthPipeline
    
    print("\n" + "="*80)
    print("DEMO: CRACK GROWTH PREDICTION WITH TRAINED MODELS")
    print("="*80)
    
    # Check if models exist
    seg_model_path = os.path.join(config.MODELS_DIR, 'unet_best.pth')
    pred_model_path = os.path.join(config.MODELS_DIR, 'growth_predictor.pkl')
    
    if not os.path.exists(seg_model_path) or not os.path.exists(pred_model_path):
        print("\n❌ ERROR: Models not found!")
        print("Please run training first:")
        print("  1. python scripts/generate_synthetic_data.py")
        print("  2. python scripts/train_pipeline.py")
        return
    
    # Create demo image
    image_path, true_mask = create_demo_crack_image('/home/claude/demo_crack.jpg')
    
    # Initialize pipeline
    print("\nInitializing prediction pipeline...")
    pipeline = CrackGrowthPipeline(seg_model_path, pred_model_path)
    
    # Test different scenarios
    scenarios = [
        {
            'name': 'High Risk - Old Outdoor Wall',
            'params': {
                'brick_type': 'clay',
                'wall_age': 45,
                'mortar_type': 'lime',
                'exposure': 'outdoor',
                'humidity': 'high',
                'load_bearing': True
            }
        },
        {
            'name': 'Low Risk - New Indoor Wall',
            'params': {
                'brick_type': 'engineering_brick',
                'wall_age': 3,
                'mortar_type': 'cement',
                'exposure': 'indoor',
                'humidity': 'low',
                'load_bearing': False
            }
        },
        {
            'name': 'Medium Risk - Moderate Conditions',
            'params': {
                'brick_type': 'concrete',
                'wall_age': 20,
                'mortar_type': 'cement_lime_mix',
                'exposure': 'outdoor',
                'humidity': 'medium',
                'load_bearing': True
            }
        }
    ]
    
    results_list = []
    
    for scenario in scenarios:
        print("\n" + "-"*80)
        print(f"SCENARIO: {scenario['name']}")
        print("-"*80)
        
        # Run prediction
        results = pipeline.predict(
            image_path=image_path,
            **scenario['params'],
            output_dir='/home/claude/demo_outputs'
        )
        
        results_list.append({
            'scenario': scenario['name'],
            'results': results['predictions']
        })
    
    # Compare scenarios
    print("\n" + "="*80)
    print("SCENARIO COMPARISON")
    print("="*80)
    
    print(f"\n{'Scenario':<35} {'Length (mm)':<15} {'Width (mm)':<15} {'Risk':<10} {'Confidence'}")
    print("-"*90)
    
    for item in results_list:
        pred = item['results']
        print(f"{item['scenario']:<35} "
              f"{pred['length_growth_mm']:>10.2f} mm    "
              f"{pred['width_growth_mm']:>10.3f} mm    "
              f"{pred['risk_label']:<10} "
              f"{pred['confidence']:>6.1%}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: /home/claude/demo_outputs/")
    print("Check the visualization images to see predicted crack growth.")


def demo_without_trained_models():
    """
    Demo showing individual components (doesn't require trained models)
    """
    print("\n" + "="*80)
    print("DEMO: INDIVIDUAL COMPONENT TESTING (NO TRAINED MODELS NEEDED)")
    print("="*80)
    
    from src.preprocessing.geometry_extraction import CrackGeometryExtractor, visualize_geometry
    from src.preprocessing.feature_engineering import FeatureEngineer
    from src.simulation.crack_growth_sim import CrackGrowthSimulator, mm_to_pixels
    from src.visualization.overlay import CrackVisualizer
    
    # Create demo crack
    print("\n[1/5] Creating demo crack...")
    wall = create_synthetic_wall_texture(size=(512, 512))
    crack_mask = create_synthetic_crack(size=(512, 512), complexity='medium')
    image_with_crack = add_crack_to_wall(wall, crack_mask)
    
    # Extract geometry
    print("[2/5] Extracting crack geometry...")
    extractor = CrackGeometryExtractor()
    geom = extractor.extract_all_features(crack_mask)
    geom['morphological'] = extractor.get_morphological_features(crack_mask)
    
    print(f"  ✓ Crack length: {geom['length_mm']:.2f} mm")
    print(f"  ✓ Average width: {geom['avg_width_mm']:.2f} mm")
    print(f"  ✓ Endpoints: {geom['num_endpoints']}")
    
    # Engineer features
    print("[3/5] Engineering features...")
    engineer = FeatureEngineer()
    features = engineer.create_features(
        geometry_features=geom,
        brick_type='clay',
        wall_age=30,
        mortar_type='cement',
        exposure='outdoor',
        humidity='high',
        load_bearing=True
    )
    print(f"  ✓ Created {len(features)} features")
    
    # Simulate growth (using dummy predictions)
    print("[4/5] Simulating crack growth...")
    simulator = CrackGrowthSimulator()
    
    # Dummy prediction: 15mm length, 0.4mm width
    growth_mask = simulator.simulate_growth(
        original_mask=crack_mask,
        skeleton=geom['skeleton'],
        endpoints=geom['endpoints'],
        directions=geom['directions'],
        predicted_length_growth_px=mm_to_pixels(15),
        predicted_width_growth_px=mm_to_pixels(0.4)
    )
    print(f"  ✓ Simulated growth complete")
    
    # Visualize
    print("[5/5] Creating visualization...")
    viz = CrackVisualizer()
    
    dummy_predictions = {
        'length_growth_mm': 15.0,
        'width_growth_mm': 0.4,
        'risk_label': 'High',
        'confidence': 0.85
    }
    
    overlay = viz.create_overlay(
        original_image=image_with_crack,
        existing_crack_mask=crack_mask,
        predicted_growth_mask=growth_mask,
        predictions=dummy_predictions
    )
    
    side_by_side = viz.create_side_by_side(
        original_image=image_with_crack,
        existing_crack_mask=crack_mask,
        predicted_growth_mask=growth_mask,
        predictions=dummy_predictions
    )
    
    # Save
    viz.save_visualization(overlay, '/home/claude/demo_overlay.png')
    viz.save_visualization(side_by_side, '/home/claude/demo_comparison.png')
    
    # Visualize geometry
    geom_viz = visualize_geometry(crack_mask * 255, geom)
    cv2.imwrite('/home/claude/demo_geometry.png', geom_viz)
    
    print("\n✓ Demo complete!")
    print("\nGenerated files:")
    print("  - demo_overlay.png       : Crack growth visualization")
    print("  - demo_comparison.png    : Side-by-side comparison")
    print("  - demo_geometry.png      : Geometry extraction visualization")


def print_usage_instructions():
    """
    Print instructions for running the demo
    """
    print("\n" + "="*80)
    print("CRACK GROWTH PREDICTION SYSTEM - DEMO SCRIPT")
    print("="*80)
    print("\nThis demo script can run in two modes:")
    print("\n1. WITH TRAINED MODELS (full pipeline):")
    print("   - Requires training first:")
    print("     $ python scripts/generate_synthetic_data.py")
    print("     $ python scripts/train_pipeline.py")
    print("   - Then run:")
    print("     $ python scripts/demo.py --mode full")
    print("\n2. WITHOUT TRAINED MODELS (component testing):")
    print("   - No training required")
    print("   - Tests individual components")
    print("   - Run:")
    print("     $ python scripts/demo.py --mode components")
    print("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo crack growth prediction system')
    parser.add_argument('--mode', type=str, default='components',
                       choices=['full', 'components'],
                       help='Demo mode: full (with trained models) or components (no models needed)')
    
    args = parser.parse_args()
    
    print_usage_instructions()
    
    if args.mode == 'full':
        demo_with_trained_models()
    else:
        demo_without_trained_models()
