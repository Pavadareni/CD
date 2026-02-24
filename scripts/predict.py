"""
Main prediction script - End-to-end inference
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch
from PIL import Image

import config
from models.segmentation.unet import UNet
from models.prediction.growth_model import CrackGrowthPredictor
from src.preprocessing.geometry_extraction import CrackGeometryExtractor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.simulation.crack_growth_sim import CrackGrowthSimulator, mm_to_pixels
from src.visualization.overlay import CrackVisualizer


# =========================================================
# PIPELINE
# =========================================================

class CrackGrowthPipeline:

    def __init__(self, segmentation_model_path, prediction_model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading segmentation model...")
        self.seg_model = self._load_segmentation_model(segmentation_model_path)

        print("Loading prediction model...")
        self.pred_model = CrackGrowthPredictor()
        self.pred_model.load(prediction_model_path)

        self.geom_extractor = CrackGeometryExtractor()
        self.feat_engineer = FeatureEngineer()
        self.simulator = CrackGrowthSimulator()
        self.visualizer = CrackVisualizer()

        print("✓ Models loaded successfully!\n")

    # =====================================================

    def _load_segmentation_model(self, model_path):

        model = UNet(
            in_channels=config.UNET_CONFIG["in_channels"],
            out_channels=config.UNET_CONFIG["out_channels"],
            init_features=config.UNET_CONFIG["init_features"]
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)   # state_dict only
        model.eval()

        return model

    # =====================================================

    def preprocess_image(self, image_path):

        image = Image.open(image_path).convert("RGB")
        image = image.resize(config.IMAGE_SIZE)
        image_np = np.array(image)

        image = image_np.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        image_tensor = torch.from_numpy(image)\
            .permute(2, 0, 1)\
            .unsqueeze(0)\
            .float()\
            .to(self.device)

        return image_tensor, image_np

    # =====================================================

    def segment_crack(self, image_tensor):

        with torch.no_grad():
            pred = self.seg_model(image_tensor)

        pred = pred.squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8)

        return mask

    # =====================================================

    def predict(
        self,
        image_path,
        brick_type,
        wall_age,
        mortar_type,
        exposure,
        humidity,
        load_bearing,
        output_dir=None
    ):

        print("=" * 80)
        print("CRACK GROWTH PREDICTION PIPELINE")
        print("=" * 80)

        # 1. Segmentation
        print("\n[1/6] Segmenting crack...")
        image_tensor, image_np = self.preprocess_image(image_path)
        crack_mask = self.segment_crack(image_tensor)
        print(f"✓ Crack detected: {np.sum(crack_mask)} pixels")

        # 2. Geometry
        print("\n[2/6] Extracting geometry...")
        geom = self.geom_extractor.extract_all_features(crack_mask)
        geom["morphological"] = self.geom_extractor.get_morphological_features(crack_mask)

        # 3. Feature engineering
        print("\n[3/6] Engineering features...")
        features = self.feat_engineer.create_features(
            geometry_features=geom,
            brick_type=brick_type,
            wall_age=wall_age,
            mortar_type=mortar_type,
            exposure=exposure,
            humidity=humidity,
            load_bearing=load_bearing
        )

        feature_vector = np.array([features[k] for k in sorted(features.keys())])

        # 4. Prediction
        print("\n[4/6] Predicting growth...")
        preds = self.pred_model.predict(feature_vector.reshape(1, -1))

        length_mm = preds["length_growth_mm"][0]
        width_mm = preds["width_growth_mm"][0]
        risk = preds["risk_label"][0]
        conf = preds["confidence"][0]

        print(f"Length growth: {length_mm:.2f} mm")
        print(f"Width growth : {width_mm:.3f} mm")
        print(f"Risk level  : {risk}")
        print(f"Confidence  : {conf:.2%}")

        # 5. Simulation
        print("\n[5/6] Simulating growth...")
        growth_mask = self.simulator.simulate_growth(
            original_mask=crack_mask,
            skeleton=geom["skeleton"],
            endpoints=geom["endpoints"],
            directions=geom["directions"],
            predicted_length_growth_px=mm_to_pixels(length_mm),
            predicted_width_growth_px=mm_to_pixels(width_mm)
        )

        # 6. Visualization
        print("\n[6/6] Creating visualization...")

        pred_dict = {
            "length_growth_mm": length_mm,
            "width_growth_mm": width_mm,
            "risk_label": risk,
            "confidence": conf
        }

        overlay = self.visualizer.create_overlay(
            image_np,
            crack_mask,
            growth_mask,
            pred_dict
        )

        side = self.visualizer.create_side_by_side(
            image_np,
            crack_mask,
            growth_mask,
            pred_dict
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]

            self.visualizer.save_visualization(
                overlay,
                os.path.join(output_dir, f"{base}_prediction.png")
            )

            self.visualizer.save_visualization(
                side,
                os.path.join(output_dir, f"{base}_comparison.png")
            )

            print(f"\nSaved outputs to {output_dir}")

        print("\nPREDICTION COMPLETE!")
        return pred_dict


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--brick_type", required=True, choices=config.BRICK_TYPES)
    parser.add_argument("--wall_age", required=True, type=int)
    parser.add_argument("--mortar_type", required=True, choices=config.MORTAR_TYPES)
    parser.add_argument("--exposure", required=True, choices=config.EXPOSURE_TYPES)
    parser.add_argument("--humidity", required=True, choices=config.HUMIDITY_LEVELS)
    parser.add_argument("--load_bearing", required=True, choices=["true", "false"])

    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seg_model", default=os.path.join(config.MODELS_DIR, "unet_best.pth"))
    parser.add_argument("--pred_model", default=os.path.join(config.MODELS_DIR, "growth_predictor.pkl"))

    args = parser.parse_args()

    load_bearing = args.load_bearing.lower() == "true"

    if not os.path.exists(args.seg_model):
        print("Segmentation model not found:", args.seg_model)
        return

    if not os.path.exists(args.pred_model):
        print("Prediction model not found:", args.pred_model)
        return

    pipeline = CrackGrowthPipeline(args.seg_model, args.pred_model)

    pipeline.predict(
        image_path=args.image,
        brick_type=args.brick_type,
        wall_age=args.wall_age,
        mortar_type=args.mortar_type,
        exposure=args.exposure,
        humidity=args.humidity,
        load_bearing=load_bearing,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()