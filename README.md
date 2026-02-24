# Wall Crack Growth Prediction System

## System Overview
A complete end-to-end AI system that predicts wall crack growth over 1 year and visualizes the predicted growth on the original input image.

## Architecture Components

```
Input Image + Metadata
        ↓
┌───────────────────────────────────────────┐
│   1. Crack Segmentation (U-Net)           │
│      - Pixel-level crack detection        │
└───────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────┐
│   2. Geometry Extraction (OpenCV)         │
│      - Skeletonization                    │
│      - Endpoint detection                 │
│      - Width estimation                   │
│      - Direction vectors                  │
└───────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────┐
│   3. Feature Engineering                  │
│      - Crack morphology features          │
│      - Structural metadata                │
│      - Environmental factors              │
└───────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────┐
│   4. Growth Prediction (XGBoost)          │
│      - Length growth (mm)                 │
│      - Width growth (mm)                  │
│      - Risk classification                │
└───────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────┐
│   5. Crack Growth Simulation              │
│      - Geometric extension from endpoints │
│      - Width expansion                    │
│      - Direction-aware propagation        │
└───────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────┐
│   6. Visualization Overlay                │
│      - Original image preserved           │
│      - Existing crack (original color)    │
│      - Predicted growth (red overlay)     │
└───────────────────────────────────────────┘
        ↓
    Output Image + Metrics
```

## Project Structure

```
crack_growth_prediction/
├── README.md
├── requirements.txt
├── config.py
├── data/
│   ├── raw/
│   │   ├── images/           # Input crack images
│   │   └── masks/            # Ground truth segmentation masks
│   ├── processed/
│   │   └── features.csv      # Extracted features + metadata
│   └── synthetic/            # Mock data for testing
├── models/
│   ├── segmentation/
│   │   ├── unet.py
│   │   └── train_segmentation.py
│   ├── prediction/
│   │   ├── growth_model.py
│   │   └── train_prediction.py
│   └── saved/                # Saved model weights
├── src/
│   ├── preprocessing/
│   │   ├── geometry_extraction.py
│   │   └── feature_engineering.py
│   ├── simulation/
│   │   └── crack_growth_sim.py
│   └── visualization/
│       └── overlay.py
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── train_pipeline.py
│   └── predict.py
└── notebooks/
    └── demo.ipynb
```

## Key Design Decisions

### 1. **U-Net for Segmentation** (vs YOLOv8-seg)
- **Why U-Net**: 
  - Superior pixel-level precision for thin crack structures
  - Skip connections preserve fine details (critical for crack width)
  - More control over architecture for civil engineering domain
  - Better for medical/structural imaging with limited data
- **Why not YOLO**: Object detection focus, less precise pixel boundaries

### 2. **XGBoost for Growth Prediction** (vs Neural Network)
- **Why XGBoost**:
  - Excellent with structured tabular data (brick type, humidity, etc.)
  - Handles feature interactions (e.g., outdoor + high humidity)
  - More interpretable for engineering decisions
  - Requires less training data
  - Built-in feature importance

### 3. **Geometry-based Growth Simulation** (vs Generative Models)
- **Why Geometry-based**:
  - Physically grounded: cracks propagate from stress concentrations (endpoints)
  - Preserves original image texture and lighting
  - Deterministic and explainable
  - No hallucinated cracks
- **Why not GANs/Diffusion**: Unpredictable, may generate unrealistic patterns

## Civil Engineering Assumptions

1. **Crack Propagation Physics**:
   - Cracks extend from endpoints (stress concentration zones)
   - Growth follows existing crack direction ± small deviation
   - Width increases uniformly along crack length

2. **Material Behavior**:
   - Clay bricks: Higher porosity → more moisture → faster growth
   - Concrete: More brittle, sudden propagation
   - Fly ash: Better durability, slower growth

3. **Environmental Factors**:
   - Outdoor + high humidity = accelerated growth (freeze-thaw cycles)
   - Load-bearing walls: stress-induced propagation
   - Wall age: older walls have material fatigue

4. **Time Scale**:
   - 1-year prediction assumes continuous exposure
   - Linear growth assumption (simplified, but reasonable for engineering estimate)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate synthetic training data
python scripts/generate_synthetic_data.py

# 2. Train the complete pipeline
python scripts/train_pipeline.py

# 3. Run prediction on a new image
python scripts/predict.py --image path/to/crack.jpg \
    --brick_type clay \
    --wall_age 25 \
    --mortar_type cement \
    --exposure outdoor \
    --humidity high \
    --load_bearing true
```

## Output Example

**Numeric Predictions**:
```
Predicted crack length growth: 12.5 mm
Predicted crack width growth: 0.3 mm
Risk category: High
Confidence score: 0.87
```

**Visual Output**:
- Original image background (preserved)
- Existing crack (as-is)
- Predicted growth (red overlay with transparency)

## System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 2GB disk space

## License
MIT License - For engineering decision support only
