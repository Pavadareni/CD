# Crack Growth Prediction System - Complete Architecture

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  • RGB crack image (512x512)                                                │
│  • Structured metadata:                                                     │
│    - Brick type (clay, concrete, fly_ash, sand_lime, engineering_brick)     │
│    - Wall age (years)                                                       │
│    - Mortar type (cement, lime, cement_lime_mix)                            │
│    - Exposure (indoor, outdoor)                                             │
│    - Humidity level (low, medium, high)                                     │
│    - Load-bearing (true/false)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 1: CRACK SEGMENTATION (U-Net)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Architecture:                                                              │
│    Encoder: 4 downsampling blocks (32→64→128→256→512 channels)              │
│    Bottleneck: 512 channels at 32x32 resolution                             │
│    Decoder: 4 upsampling blocks with skip connections                       │
│                                                                             │
│  Loss Function: Combined Dice (0.7) + BCE (0.3)                             │
│                                                                             │
│  Output: Binary mask (512x512) where 1 = crack, 0 = background              │
│                                                                             │
│  Why U-Net?                                                                 │
│    • Skip connections preserve thin crack details                           │
│    • Proven for structural/medical imaging                                  │
│    • Works with limited training data                                       │
│    • Better pixel-level precision than YOLO                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           MODULE 2: GEOMETRY EXTRACTION (OpenCV + scikit-image)             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 2.1: Skeletonization                                                  │
│    • Morphological thinning → 1-pixel centerline                            │
│    • Preserves topology and connectivity                                    │
│                                                                             │
│  Step 2.2: Endpoint Detection                                               │
│    • Identify pixels with exactly 1 neighbor                                │
│    • These are stress concentration points                                  │
│                                                                             │
│  Step 2.3: Width Estimation                                                 │
│    • Distance transform from mask                                           │
│    • Width = 2 × distance at skeleton pixels                                │
│                                                                             │
│  Step 2.4: Direction Vectors                                                │
│    • Fit line to nearby skeleton pixels                                     │
│    • Direction points away from crack body                                  │
│                                                                             │
│  Output:                                                                    │
│    • Skeleton (binary mask)                                                 │
│    • Endpoints [(y, x), ...]                                                │
│    • Direction vectors [(dy, dx), ...]                                      │
│    • Width map (pixels)                                                     │
│    • Length (mm), average width (mm), area (mm²)                            │
│    • Morphological features (eccentricity, solidity, etc.)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  MODULE 3: FEATURE ENGINEERING                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Feature Categories (~40 features):                                         │
│                                                                             │
│  1. GEOMETRIC (from image):                                                 │
│     • Crack length, width, area                                             │
│     • Number of endpoints                                                   │
│     • Perimeter-to-area ratio (tortuosity)                                  │
│     • Eccentricity, solidity                                                │
│                                                                             │
│  2. MATERIAL PROPERTIES:                                                    │
│     • Brick durability factor (engineering coefficient)                     │
│     • Mortar strength factor                                                │
│     • One-hot encoded brick type (5 features)                               │
│     • One-hot encoded mortar type (3 features)                              │
│                                                                             │
│  3. AGE & DEGRADATION:                                                      │
│     • Wall age (years)                                                      │
│     • Wall age squared (non-linear aging)                                   │
│     • Age categories (new, medium, old)                                     │
│                                                                             │
│  4. ENVIRONMENTAL:                                                          │
│     • Exposure multiplier (outdoor = 1.5× indoor)                           │
│     • Humidity multiplier (high = 1.6× low)                                 │
│     • One-hot humidity levels (3 features)                                  │
│                                                                             │
│  5. STRUCTURAL LOAD:                                                        │
│     • Load-bearing flag                                                     │
│                                                                             │
│  6. INTERACTION FEATURES:                                                   │
│     • Severe exposure = outdoor × high humidity                             │
│     • Structural fatigue = old age × load bearing                           │
│     • Mortar-environmental risk                                             │
│     • Combined degradation factor                                           │
│                                                                             │
│  Output: Feature vector (1 × 40)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              MODULE 4: GROWTH PREDICTION (XGBoost)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Architecture: 3 separate XGBoost models                                    │
│                                                                             │
│  Model 4.1: Length Growth Regression                                        │
│    • Objective: reg:squarederror                                            │
│    • Predicts: Crack length increase (mm) over 1 year                       │
│                                                                             │
│  Model 4.2: Width Growth Regression                                         │
│    • Objective: reg:squarederror                                            │
│    • Predicts: Crack width increase (mm) over 1 year                        │
│                                                                             │
│  Model 4.3: Risk Classification                                             │
│    • Objective: multi:softmax (3 classes)                                   │
│    • Classes: Low (<5mm), Medium (5-15mm), High (>15mm)                     │
│                                                                             │
│  Hyperparameters:                                                           │
│    • max_depth: 6                                                           │
│    • learning_rate: 0.1                                                     │
│    • n_estimators: 100                                                      │
│                                                                             │
│  Why XGBoost?                                                               │
│    • Excellent with structured/tabular features                             │
│    • Handles non-linear interactions (brick × humidity)                     │
│    • Built-in feature importance                                            │
│    • Robust to feature scales                                               │
│    • More interpretable than neural networks                                │
│                                                                             │
│  Output:                                                                    │
│    • Length growth: 0-50 mm                                                 │
│    • Width growth: 0-2 mm                                                   │
│    • Risk category: {Low, Medium, High}                                     │
│    • Confidence score: 0.3-0.95                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│         MODULE 5: CRACK GROWTH SIMULATION (Geometry-based)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 5.1: CRACK EXTENSION (length growth)                                  │
│                                                                             │
│    For each endpoint:                                                       │
│      1. Start at endpoint coordinates                                       │
│      2. Get direction vector from geometry extraction                       │
│      3. For N steps:                                                        │
│         • Add small angular deviation (±15°) for realism                    │
│         • Move step_size pixels in direction                                │
│         • Draw line segment (thickness=2)                                   │
│         • Update position                                                   │
│      4. Result: Extended crack from tip                                     │
│                                                                             │
│    Physics rationale:                                                       │
│      • Stress concentrates at crack tips (endpoints)                        │
│      • Propagation follows existing crack direction                         │
│      • Material heterogeneity causes small deviations                       │
│                                                                             │
│  Step 5.2: CRACK WIDENING (width growth)                                    │
│                                                                             │
│    Method: Morphological dilation                                           │
│      1. Create circular kernel (radius = width_growth_px)                   │
│      2. Dilate original crack mask                                          │
│      3. Subtract original to get only new width                             │
│                                                                             │
│    Physics rationale:                                                       │
│      • Continued stress widens existing crack                               │
│      • Widening is relatively uniform                                       │
│                                                                             │
│  Step 5.3: COMBINE                                                          │
│    • Merge extension + widening masks                                       │
│    • Remove overlap with original crack                                     │
│                                                                             │
│  Why NOT GANs/Diffusion?                                                    │
│    • Unpredictable: may generate unrealistic patterns                       │
│    • Not physics-based                                                      │
│    • Cannot guarantee texture/lighting preservation                         │
│    • Less interpretable for engineering decisions                           │
│                                                                             │
│  Output: Growth mask (512x512) showing ONLY new growth                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 MODULE 6: VISUALIZATION OVERLAY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Step 6.1: Prepare Overlays                                                 │
│    • Existing crack → Green (0, 255, 0)                                     │
│    • Predicted growth → Red (255, 0, 0)                                     │
│    • Alpha blending: 0.6 transparency                                       │
│                                                                             │
│  Step 6.2: Composite Image                                                  │
│    Output = Original × (1 - mask × alpha) + Overlay × (mask × alpha)        │
│                                                                             │
│    This preserves:                                                          │
│      • Original texture                                                     │
│      • Lighting conditions                                                  │
│      • Background details                                                   │
│                                                                             │
│  Step 6.3: Add Annotations                                                  │
│    • Prediction metrics (length, width, risk)                               │
│    • Confidence score                                                       │
│    • Color legend                                                           │
│                                                                             │
│  Output Formats:                                                            │
│    1. Single overlay: Original + both cracks                                │
│    2. Side-by-side: Original | Prediction                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Visual Output:                                                             │
│    • RGB image with overlays                                                │
│    • Original texture preserved                                             │
│    • Existing crack (green) + Predicted growth (red)                        │
│    • Annotations panel                                                      │
│                                                                             │
│  Numeric Output:                                                            │
│    • Predicted crack length growth: X.XX mm                                 │
│    • Predicted crack width growth: X.XXX mm                                 │
│    • Risk category: {Low | Medium | High}                                   │
│    • Confidence score: 0.XX                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Engineering Assumptions

### Material Behavior
```
Brick Durability Factors (higher = more prone to cracking):
  • Clay brick:          1.2  (porous, moisture absorption)
  • Concrete brick:      1.0  (baseline)
  • Fly ash brick:       0.8  (better durability)
  • Sand lime brick:     1.1  
  • Engineering brick:   0.7  (highest quality)

Mortar Strength Factors (higher = weaker):
  • Cement mortar:       1.0  (strong, baseline)
  • Lime mortar:         1.5  (weaker, more flexible)
  • Cement-lime mix:     1.2  (intermediate)
```

### Environmental Effects
```
Exposure Multipliers:
  • Indoor:   1.0× (controlled environment)
  • Outdoor:  1.5× (freeze-thaw, moisture cycles)

Humidity Multipliers:
  • Low:      1.0× (minimal moisture)
  • Medium:   1.3× (moderate moisture)
  • High:     1.6× (accelerated degradation)
```

### Crack Propagation Physics
```
1. Stress Concentration:
   • Maximum stress at crack tips (endpoints)
   • Growth initiates from these points

2. Direction Continuity:
   • Cracks extend in existing direction
   • Small deviations (±15°) due to material grain

3. Time-Dependent Growth:
   • Linear assumption over 1 year
   • Actual: cyclic (seasonal) + cumulative

4. Width Evolution:
   • Uniform expansion along crack length
   • Driven by continuing stress
```

## Model Selection Rationale

### U-Net vs YOLOv8-seg
```
CHOSEN: U-Net

Advantages:
  ✓ Superior pixel-level precision
  ✓ Skip connections preserve thin cracks
  ✓ Works with limited data
  ✓ More control over architecture

YOLOv8-seg disadvantages:
  ✗ Object detection focus
  ✗ Less precise boundaries
  ✗ Requires more data
```

### XGBoost vs Neural Network
```
CHOSEN: XGBoost

Advantages:
  ✓ Excellent with tabular features
  ✓ Handles feature interactions naturally
  ✓ More interpretable (feature importance)
  ✓ Less prone to overfitting
  ✓ Requires less training data

Neural network disadvantages:
  ✗ Needs more data
  ✗ Black box
  ✗ Harder to tune
```

### Geometry-based vs Generative
```
CHOSEN: Geometry-based simulation

Advantages:
  ✓ Physically grounded
  ✓ Preserves original image
  ✓ Deterministic and explainable
  ✓ No hallucinated cracks

GAN/Diffusion disadvantages:
  ✗ Unpredictable results
  ✗ May alter lighting/texture
  ✗ Not physics-based
  ✗ Requires large datasets
```

## Performance Considerations

### Computation Time (estimated)
```
Segmentation (U-Net):        ~0.1 seconds (GPU) / ~2 seconds (CPU)
Geometry Extraction:         ~0.05 seconds
Feature Engineering:         ~0.01 seconds
Growth Prediction (XGBoost): ~0.02 seconds
Crack Simulation:            ~0.1 seconds
Visualization:               ~0.05 seconds
──────────────────────────────────────────────────
Total Pipeline:              ~0.3 seconds (GPU) / ~2.2 seconds (CPU)
```

### Memory Requirements
```
Model Weights:
  • U-Net: ~50 MB
  • XGBoost: ~2 MB

Runtime Memory:
  • Peak: ~2 GB (during segmentation)
  • Minimum: 4 GB RAM recommended
```

## Limitations & Future Work

### Current Limitations
```
1. Linear growth assumption (actual: non-linear over time)
2. No seasonal/cyclic effects modeled
3. Single crack focus (no multi-crack interaction)
4. 2D analysis (real cracks are 3D)
5. Calibration required (pixel-to-mm conversion)
```

### Potential Improvements
```
1. Time-series model for non-linear growth
2. Multi-scale segmentation for thin cracks
3. 3D crack modeling (depth estimation)
4. Ensemble predictions for uncertainty quantification
5. Active learning from expert feedback
6. Integration with structural analysis software
```

## Use Cases

### Engineering Decision Support
```
✓ Maintenance prioritization
✓ Repair vs replacement decisions
✓ Budget forecasting
✓ Risk assessment
✓ Safety evaluation
```

### NOT Suitable For
```
✗ Life-safety critical decisions (use as aid only)
✗ Legal/insurance claims (requires expert validation)
✗ Precise structural calculations (simplified model)
```
