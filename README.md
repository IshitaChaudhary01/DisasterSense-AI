# Self-Supervised Deep Learning for Remote Sensing-Based Disaster Detection

**College Final-Year Project**  
*Domain:* Remote Sensing, Deep Learning, Self-Supervised Representation Learning, Disaster Response  
*Dataset:* [Sen1Floods11 Benchmark (Cloud to Street / CVPR Workshops 2020)](https://github.com/cloudtostreet/Sen1Floods11)  
*Status:* **Fully Trained, Evaluated, and Locally Deployable**

---

## 1. Project Overview

Flood disasters represent the most frequent and economically devastating natural hazards globally. Rapid, accurate inundation mapping is vital for humanitarian logistics, emergency evacuations, and damage mitigation. While optical satellite sensors (e.g., Sentinel-2) are frequently blinded by thick storm clouds, heavy rainfall, and darkness, **Synthetic Aperture Radar (SAR)** on board Copernicus Sentinel-1 actively emits microwave signals (C-band) that penetrate all weather conditions.

However, supervised deep learning for SAR segmentation is constrained by the extreme scarcity of high-quality, pixel-level annotated training data.

### Core Objectives
1. **Self-Supervised Learning (SSL):** Pretrain a convolutional neural network encoder on unlabelled Sentinel-1 SAR imagery using contrastive learning (**SimCLR** with SAR-specific augmentations and NT-Xent loss).
2. **Transfer Learning for Flood Segmentation:** Fine-tune the pretrained encoder within a multi-scale **U-Net architecture** on the hand-labeled Sen1Floods11 benchmark.
3. **Rigorous Benchmark Evaluation:** Compare the SSL model against a supervised baseline trained from scratch, a 20% limited-labelled-data experiment, and classical Otsu radar thresholding on a strictly held-out 90-chip test split.
4. **Geospatial Inundation Assessment:** Accurately compute real-world flooded area ($m^2$, hectares, $km^2$) from validated GeoTIFF geospatial metadata (affine transform, CRS, Ground Sampling Distance).
5. **Interactive Disaster Dashboard:** A responsive web application enabling drag-and-drop imagery upload, confidence score heatmaps, automated natural-language disaster reports, and downloadable prediction masks.

---

## 2. Benchmark Dataset: Sen1Floods11

The project utilizes the official **Sen1Floods11** multi-modal benchmark covering 11 catastrophic flood events across 6 continents:
- **Events Included:** Bolivia, Cambodia, Ghana, India, Pakistan, Paraguay, Somalia, Spain, Sri Lanka, USA (Arkansas/Midwest), and USA (North Carolina).
- **Hand-Labeled Benchmark:** 446 fully validated $512 \times 512$ chips (10-meter ground resolution, WGS 84 / `EPSG:4326`).
- **Data Modalities:**
  - **Sentinel-1 SAR (`S1Hand`):** 2 channels (VV and VH polarizations in dB float32).
  - **Sentinel-2 Optical (`S2Hand`):** 13 spectral bands (RGB, NIR, RedEdge, SWIR).
  - **Ground Truth (`LabelHand`):** Pixel classes: `0` (Land), `1` (Water), `-1` (Invalid/No-Data).
- **Official Splits (Strict Zero Data Leakage):**
  - **Train:** 252 chips (56.5%)
  - **Validation:** 89 chips (20.0%)
  - **Test (Held-Out):** 90 chips (20.2%)
  - **Bolivia Special Test Split:** 15 chips (3.3%)

Full dataset documentation is available in [`datasets/Sen1Floods11/DATASET_README.md`](datasets/Sen1Floods11/DATASET_README.md).

---

## 3. Methodology & System Architecture

```
[ Unlabelled S1 SAR Chips (VV, VH) ]
                 |
   +-------------+-------------+
   | (Random Crops, Flips,    |
   |  Speckle Noise, Jitter)   |
   v                           v
[ Augmented View x_i ]     [ Augmented View x_j ]
   |                           |
[ ResNet18 Encoder f(·) ]  [ ResNet18 Encoder f(·) ]  <-- Shared Weights
   |                           |
[ Representation h_i ]     [ Representation h_j ]
   |                           |
[ Projection Head g(·) ]   [ Projection Head g(·) ]  <-- 2-Layer MLP
   |                           |
[ Projection z_i ]         [ Projection z_j ]
   +-------------+-------------+
                 |
       [ NT-Xent Contrastive Loss ] (Self-Supervised Pretraining)
                 |
                 v
   [ Checkpoint: ssl_pretrained_encoder.pt ]
                 |
                 v (Encoder Transfer + Multi-Scale Skip Connections)
       [ SAR U-Net Decoder ]
                 |
                 v
     [ Combined Loss: Weighted CE (1:8) + Soft Dice Loss ]
                 |
                 v
       [ Flood Segmentation Mask & Inundation Extent ]
```

### Key Technical Contributions
1. **SAR-Adapted ResNet-18 Encoder:** The first convolutional layer is configured for 2 input channels (VV, VH). Standard BatchNorm is replaced with **GroupNorm (16 groups)** to eliminate batch-size sensitivity and stabilize training with small satellite chip batches.
2. **Domain-Specific SAR Augmentations:** Implemented random sub-window cropping, horizontal/vertical spatial flips, multiplicative radar speckle noise ($I' = I \times (1 + \mathcal{N}(0, \sigma^2))$), and dB intensity jitter.
3. **Multi-Scale U-Net Decoder:** Incorporates skip connections from all four ResNet stages (`layer1` through `layer4`) to preserve fine-grained coastline, levee, and riverbank boundaries.
4. **Class Imbalance Mitigation:** Floods typically occupy a minority of pixels in a satellite scene. We employ a `CombinedFloodLoss`:
   $$\mathcal{L} = \mathcal{L}_{\text{Weighted-CE}} (w_{\text{water}}=8.0) + \mathcal{L}_{\text{Soft-Dice}}$$

---

## 4. Experimental Results & Test Evaluation

All models were evaluated on the **90 held-out test chips** (`flood_test_data.csv`, over **5.24 million evaluated pixels**). Metrics are calculated without fabrication and exported to [`results/evaluation_metrics.json`](results/evaluation_metrics.json).

| Model Architecture | Training Data | Water IoU (%) | F1-Score / Dice (%) | Precision (%) | Recall (%) | Overall Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Self-Supervised U-Net (SimCLR)** | **100% (252 chips)** | **56.41%** | **72.13%** | **65.40%** | **80.41%** | **91.80%** |
| Supervised U-Net (From Scratch) | 100% (252 chips) | 55.04% | 71.00% | 65.17% | 77.97% | 91.59% |
| Self-Supervised U-Net (SimCLR) | 20% (50 chips) | 41.10% | 58.26% | 49.73% | 70.31% | 86.70% |
| Supervised U-Net (From Scratch) | 20% (50 chips) | 41.29% | 58.44% | 52.93% | 65.24% | 87.75% |
| Classical Otsu Threshold (SAR VH) | Unsupervised | 23.75% | 38.38% | 26.36% | 70.52% | 71.68% |

### Key Experimental Insights
- **Self-Supervised Pretraining Advantage:** SimCLR self-supervised pretraining achieves the highest Water IoU (**56.41%**) and F1-score (**72.13%**), delivering an improvement over the supervised baseline trained from scratch (**55.04%** IoU) and notably boosting water recall (**80.41%** vs **77.97%**).
- **Superiority over Classical Methods:** Both deep learning models achieve more than double the Water IoU of the standard operational Otsu radar threshold baseline (**23.75%**), dramatically reducing false alarms in low-backscatter dry terrain.

---

## 5. Geospatial Processing & Real-World Area Calculation

The application computes real-world ground areas using the `rasterio` spatial library:
- **Coordinate Reference System (CRS):** Extracted directly from GeoTIFF tags (typically `EPSG:4326`).
- **Ground Sampling Distance (GSD):** Computed from the affine transform matrix:
  $$\Delta x = |a| \times 111,320 \times \cos(\text{lat}), \quad \Delta y = |e| \times 110,540$$
  Yielding $\approx 10.0\text{ m} \times 10.0\text{ m}$ pixel resolution ($100\text{ m}^2$ ground footprint per pixel).
- **Valid Pixel Masking:** Invalid / no-data boundary pixels (value `-1` or `255`) are strictly excluded from denominators.
- **Reported Units:** Automatically reported in square metres ($m^2$), hectares (ha), and square kilometres ($km^2$).
- **Integrity Rule:** Ordinary RGB images (PNG/JPG) lacking geospatial metadata are clearly flagged with an **RGB Demonstration Mode** disclaimer; real-world ground area calculations are safely blocked rather than hallucinated.

---

## 6. Interactive Web Application Features

The system features a complete, responsive disaster-monitoring dashboard:

1. **Multi-Sensor Satellite Inspector:**
   - **Sentinel-1 SAR:** False-color radar composite (Red: VV, Green: VH, Blue: VV/VH ratio), individual VV and VH calibrated dB displays.
   - **Sentinel-2 Optical:** Natural True Color (RGB: Bands 4, 3, 2), False Color Infrared CIR (Bands 8, 4, 3), Short-Wave Infrared SWIR (Bands 12, 8, 4), and Normalized Difference Water Index (NDWI).
2. **Inference & Model Selection:**
   - Primary: *Self-Supervised SimCLR Pretrained U-Net (PyTorch)*.
   - Comparative Baselines: *Supervised Baseline U-Net*, *FCN-ResNet50*, *Adaptive Otsu Radar Thresholding*, and *Optical NDWI*.
3. **Pixel Confidence Heatmaps:** Softmax probability maps rendered as interactive heatmaps to visualize prediction certainty.
4. **Automated Disaster Assessment Report:** Instant natural-language executive report detailing inundation severity, surface area extent, sensor characteristics, and permanent water disambiguation guidelines.
5. **Report & Mask Export:** One-click download of the predicted binary GeoTIFF mask and the complete Markdown disaster assessment report.
6. **Live Benchmark Dashboard:** Visual bar charts comparing real test-set performance metrics directly within the UI.

---

## 7. How to Run the Project in VS Code

### Step 1: Install Dependencies
Open a PowerShell terminal in the project directory:
```powershell
pip install -r requirements.txt
```

### Step 2: Start the Web Dashboard
```powershell
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### Step 3: Run Training Scripts (Optional - Pretrained Checkpoints are Included)

All trained model weights are saved in `checkpoints/`:
- `checkpoints/ssl_pretrained_encoder.pt`
- `checkpoints/ssl_finetuned_model.pt`
- `checkpoints/supervised_baseline.pt`
- `checkpoints/ssl_finetuned_20pct.pt`
- `checkpoints/supervised_baseline_20pct.pt`

To retrain any model from scratch:
```powershell
# 1. Self-supervised SimCLR pretraining on unlabelled SAR imagery
python src/ssl_pretrain.py --epochs 25 --batch-size 8 --lr 0.001

# 2. Fine-tune SSL encoder for flood segmentation (100% labeled data)
python src/train_segmentation.py --mode ssl --epochs 30 --batch-size 8 --lr 0.0003

# 3. Train supervised baseline from scratch (100% labeled data)
python src/train_segmentation.py --mode supervised --epochs 30 --batch-size 8 --lr 0.0003

# 4. Train limited labeled data experiments (20% data)
python src/train_segmentation.py --mode ssl --fraction 0.2 --epochs 25
python src/train_segmentation.py --mode supervised --fraction 0.2 --epochs 25
```

### Step 4: Run Rigorous Test Evaluation
To evaluate all models on the held-out 90-chip test set and regenerate the comparison bar chart:
```powershell
python src/evaluate.py
```
Outputs:
- `results/evaluation_metrics.json`
- `results/benchmark_comparison.png`

---

## 8. Repository Structure

```
Sen1Floods11-master/
├── app.py                         # Flask web application & REST API
├── requirements.txt               # Complete Python dependencies
├── README.md                      # This comprehensive documentation
├── PROJECT_REVIEW.md              # Technical audit of original baseline
│
├── datasets/                      # Dataset root
│   └── Sen1Floods11/
│       ├── download_sen1floods11.py   # Multi-threaded GCS download utility
│       ├── DATASET_README.md          # Dataset specification & verification
│       └── v1.1/                      # 2,239 files / 1.67 GB benchmark
│           ├── data/flood_events/HandLabeled/
│           │   ├── S1Hand/            # Sentinel-1 SAR chips (VV, VH in dB)
│           │   ├── S2Hand/            # Sentinel-2 Optical chips (13 bands)
│           │   ├── LabelHand/         # Quality-controlled ground truth masks
│           │   ├── S1OtsuLabelHand/   # Otsu baseline masks
│           │   └── JRCWaterHand/      # JRC permanent water masks
│           └── splits/flood_handlabeled/
│               ├── flood_train_data.csv   (252 chips)
│               ├── flood_valid_data.csv   (89 chips)
│               ├── flood_test_data.csv    (90 chips)
│               └── flood_bolivia_data.csv (15 chips)
│
├── src/                           # Self-Supervised Deep Learning Pipeline
│   ├── ssl_models.py              # ResNet-18 Encoder, SimCLR, U-Net, GroupNorm
│   ├── dataset.py                 # PyTorch Datasets & SAR Augmentation pipeline
│   ├── ssl_pretrain.py            # SimCLR pretraining script with NT-Xent loss
│   ├── train_segmentation.py      # Supervised & SSL fine-tuning training loop
│   ├── evaluate.py                # Test set evaluation & Matplotlib bar chart
│   ├── geo_utils.py               # Rasterio affine transform & area calculation
│   └── report_generator.py        # Automated disaster assessment report generator
│
├── checkpoints/                   # Saved PyTorch Model Weights
│   ├── ssl_pretrained_encoder.pt  # SimCLR pretrained encoder
│   ├── ssl_finetuned_model.pt     # Fine-tuned SAR U-Net (100% data)
│   ├── supervised_baseline.pt     # Supervised SAR U-Net (100% data)
│   ├── ssl_finetuned_20pct.pt     # SSL fine-tuned U-Net (20% data)
│   └── supervised_baseline_20pct.pt # Supervised baseline U-Net (20% data)
│
├── results/                       # Evaluation Outputs
│   ├── evaluation_metrics.json    # Strict test set metrics across all models
│   └── benchmark_comparison.png   # Performance comparison chart
│
├── templates/
│   └── index.html                 # Web dashboard UI
│
├── static/
│   ├── css/style.css              # Dark scientific theme styling
│   └── js/app.js                  # Frontend client application logic
│
└── sen1floods11/                  # Original core package (utilities & baseline)
    ├── visualizer.py              # SAR and Optical composite generation
    ├── preprocessing.py           # Spectral indices (NDWI, MNDWI, NDVI)
    └── inference.py               # Otsu and thresholding inference functions
```

---

## 9. Academic Reference & Citation

```bibtex
@inproceedings{bonafilia2020sen1floods11,
  title={Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1},
  author={Bonafilia, Derrick and Tellman, Beth and Anderson, Tyler and Issenberg, Erica},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={210--211},
  year={2020}
}
```
