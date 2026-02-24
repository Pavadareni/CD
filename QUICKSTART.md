# Quick Start Guide

## Installation

```bash
# Clone or navigate to project directory
cd crack_growth_prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step-by-Step Usage

### Step 1: Generate Synthetic Training Data

Since no public crack growth dataset exists, we generate synthetic data for training:

```bash
python scripts/generate_synthetic_data.py
```

This creates:
- 500 synthetic crack images
- Corresponding segmentation masks
- Metadata with growth labels
- Saved in `data/synthetic/`

**Output:**
```
data/synthetic/
├── images/           # 500 crack images
├── masks/            # 500 binary masks
└── metadata.csv      # Labels and metadata
```

### Step 2: Train the Complete Pipeline

Train both the segmentation and prediction models:

```bash
python scripts/train_pipeline.py
```

This will:
1. Train U-Net segmentation model (~20 minutes on GPU)
2. Extract features from all images
3. Train XGBoost growth prediction model (~2 minutes)
4. Save trained models

**Output:**
```
models/saved/
├── unet_best.pth           # Trained U-Net weights
└── growth_predictor.pkl    # Trained XGBoost model
```

### Step 3: Run Prediction on New Image

#### Option A: Using Command Line

```bash
python scripts/predict.py \
    --image path/to/crack_image.jpg \
    --brick_type clay \
    --wall_age 25 \
    --mortar_type cement \
    --exposure outdoor \
    --humidity high \
    --load_bearing true \
    --output_dir /mnt/user-data/outputs
```

#### Option B: Using Python

```python
from scripts.predict import CrackGrowthPipeline

# Initialize pipeline
pipeline = CrackGrowthPipeline(
    segmentation_model_path='models/saved/unet_best.pth',
    prediction_model_path='models/saved/growth_predictor.pkl'
)

# Run prediction
results = pipeline.predict(
    image_path='path/to/crack_image.jpg',
    brick_type='clay',
    wall_age=25,
    mortar_type='cement',
    exposure='outdoor',
    humidity='high',
    load_bearing=True,
    output_dir='/mnt/user-data/outputs'
)

# Access results
print(f"Length growth: {results['predictions']['length_growth_mm']:.2f} mm")
print(f"Risk: {results['predictions']['risk_label']}")
```

**Output:**
```
/mnt/user-data/outputs/
├── crack_image_prediction.png   # Overlay visualization
└── crack_image_comparison.png   # Side-by-side comparison
```

## Example Scenarios

### Scenario 1: Old Outdoor Wall
```bash
python scripts/predict.py \
    --image examples/outdoor_crack.jpg \
    --brick_type clay \
    --wall_age 40 \
    --mortar_type lime \
    --exposure outdoor \
    --humidity high \
    --load_bearing true
```

Expected: High risk, significant growth prediction

### Scenario 2: New Indoor Wall
```bash
python scripts/predict.py \
    --image examples/indoor_crack.jpg \
    --brick_type engineering_brick \
    --wall_age 5 \
    --mortar_type cement \
    --exposure indoor \
    --humidity low \
    --load_bearing false
```

Expected: Low risk, minimal growth prediction

## Parameter Descriptions

### Required Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `--image` | string | file path | Path to crack image (JPEG/PNG) |
| `--brick_type` | string | `clay`, `concrete`, `fly_ash`, `sand_lime`, `engineering_brick` | Type of brick material |
| `--wall_age` | integer | 0-100+ | Age of wall in years |
| `--mortar_type` | string | `cement`, `lime`, `cement_lime_mix` | Type of mortar used |
| `--exposure` | string | `indoor`, `outdoor` | Environmental exposure |
| `--humidity` | string | `low`, `medium`, `high` | Humidity level |
| `--load_bearing` | string | `true`, `false` | Is wall load-bearing? |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--output_dir` | string | `/mnt/user-data/outputs` | Where to save results |
| `--seg_model` | string | `models/saved/unet_best.pth` | Segmentation model path |
| `--pred_model` | string | `models/saved/growth_predictor.pkl` | Prediction model path |

## Understanding the Output

### Visual Output

**Overlay Image:**
- Background: Original crack image (preserved)
- Green overlay: Existing crack (as detected)
- Red overlay: Predicted crack growth (1 year)
- Panel: Numeric predictions

**Side-by-Side Image:**
- Left: Original image
- Right: Prediction overlay
- Bottom: Metrics panel

### Numeric Output

```
Predicted crack length growth: 12.5 mm
Predicted crack width growth: 0.35 mm
Risk category: High
Confidence score: 87%
```

**Risk Categories:**
- **Low**: < 5 mm total growth
- **Medium**: 5-15 mm total growth  
- **High**: > 15 mm total growth

**Confidence Score:**
- 0.3-0.5: Low confidence (uncertain)
- 0.5-0.75: Medium confidence
- 0.75-0.95: High confidence

## Troubleshooting

### Problem: "Models not found"
**Solution:** Run training first:
```bash
python scripts/train_pipeline.py
```

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size in `config.py`:
```python
UNET_CONFIG = {
    'batch_size': 4,  # Reduce from 8
    ...
}
```

### Problem: "Crack not detected"
**Possible causes:**
1. Crack too small/thin (< 0.5mm width)
2. Poor image quality/lighting
3. Need to retrain on similar images

**Solution:** Adjust segmentation threshold or retrain with similar data

### Problem: Unrealistic predictions
**Causes:**
1. Out-of-distribution input (unusual brick type, extreme age)
2. Model needs more training data

**Solution:** Add more diverse training data

## Performance Optimization

### CPU-only Deployment
If no GPU available:
```python
# In config.py or environment variable
export CUDA_VISIBLE_DEVICES=""
```

Expect ~10x slower inference (~2 seconds vs 0.2 seconds)

### Batch Processing
For multiple images:
```python
import glob

images = glob.glob('path/to/images/*.jpg')
for image_path in images:
    results = pipeline.predict(
        image_path=image_path,
        # ... parameters
    )
```

## Next Steps

1. **Test with Real Data**: Try with actual crack images
2. **Calibrate**: Measure pixel-to-mm ratio for your images
3. **Fine-tune**: Add real labeled data and retrain
4. **Integrate**: Connect to your maintenance workflow

## Getting Help

- Check `ARCHITECTURE.md` for system details
- See `README.md` for overview
- Review code comments for implementation details
