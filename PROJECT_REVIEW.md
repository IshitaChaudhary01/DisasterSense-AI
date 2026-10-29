# Comprehensive Technical Review: Sen1Floods11

## 1. Executive Summary & Scientific Purpose

**Sen1Floods11** is a benchmark dataset and deep learning platform for operational flood mapping developed by Cloud to Street (Bonafilia et al., published in *IEEE/CVF CVPR Workshops 2020*). 

Flooding is the most frequent and economically devastating category of natural disasters worldwide. Traditional optical satellite monitoring (e.g., Sentinel-2 MSI, Landsat-8, MODIS) is severely limited during active flood emergencies because floods are fundamentally meteorological events accompanied by dense storm clouds, torrential rain, and atmospheric occlusions.

**Sentinel-1 Synthetic Aperture Radar (SAR)** circumvents this constraint:
- **C-band Active Microwave Sensing (5.405 GHz)**: Transmits radar pulses that penetrate cloud cover, precipitation, aerosols, and darkness.
- **Surface Roughness & Specular Reflection**: Calm, open floodwaters act as specular reflectors, scattering radar pulses away from the sensor antenna and resulting in characteristically low radar backscatter (dark pixels in VV and VH bands, typically $\le -22$ dB).
- **Dual Polarization**: Sentinel-1 measures both co-polarization (VV: Vertical transmit, Vertical receive) and cross-polarization (VH: Vertical transmit, Horizontal receive). VH is particularly sensitive to volume scattering from vegetation, while VV captures surface roughness.

---

## 2. Dataset Architecture & Specifications

| Dimension | Specification |
| :--- | :--- |
| **Ground Resolution** | 10 meters per pixel |
| **Spatial Projection** | WGS 84 (`EPSG:4326`) |
| **Chip Dimensions** | 512 × 512 pixels ($\approx 26.21\text{ km}^2$ or $2,621.4\text{ ha}$ per chip) |
| **Total Chips** | 4,831 chips across 11 flood disaster events |
| **Hand-Labeled Chips** | 446 chips (verified by trained human annotators) |
| **Weakly-Labeled Chips** | 4,385 chips (derived from Otsu thresholding & optical indices) |
| **Permanent Water Chips**| 815 chips (derived from JRC Global Surface Water dataset) |

### Sensor Modalities
1. **Sentinel-1 GRD (Ground Range Detected)**:
   - 2 bands: `Band 0: VV`, `Band 1: VH`.
   - Data format: Float32 GeoTIFF, values expressed in decibels (dB), typical backscatter range: $[-45\text{ dB}, +5\text{ dB}]$.
2. **Sentinel-2 MSI Level-1C (Multispectral)**:
   - 13 spectral bands (B1 through B12, including RedEdge, NIR, Water Vapor, and SWIR).
   - Data format: UInt16 GeoTIFF, TOA (Top of Atmosphere) reflectance scaled by 10,000.
3. **Quality Control Hand Labels (Ground Truth)**:
   - `-1`: Invalid / No Data / Clouds
   - `0`: Non-Water / Dry Land
   - `1`: Water / Flood

---

## 3. Codebase Audit: Strengths, Weaknesses, and Gaps

### Strengths
- **Rigorous Ground Truth**: 446 hand-labeled chips provide a reliable benchmark against which automated thresholding and deep learning models can be evaluated.
- **Global Geographic Heterogeneity**: 11 flood events across 5 continents (Bolivia, Colombia, Ghana, India, Cambodia, Nigeria, Pakistan, Paraguay, Somalia, Spain, USA) capture distinct biomes, soil types, urban topographies, and agricultural terrain.
- **Architectural Adaptation for High-Resolution Imagery**: The authors replaced standard `BatchNorm2d` layers with `GroupNorm(num_groups=16)`. Satellite chips ($512 \times 512 \times C$) consume significant GPU memory, capping training batch sizes to 2 or 4. At batch sizes $< 8$, BatchNorm batch statistics become noisy and degrade training; GroupNorm normalizes across channel groups per image, providing consistent convergence.

### Weaknesses & Gaps Identified in Original Repo
1. **Lack of Modular Software Architecture**:
   - The entire training pipeline was confined to Google Colab Jupyter notebooks (`Train.ipynb`, `old_training/Main_Training_Stuff.ipynb`).
   - Code was not packaged as an importable Python library or CLI.
2. **Linux & Colab Hardcoding**:
   - Code contained hardcoded Linux commands (`!apt-get install gcsfuse`, `!sudo mkdir /home/files`, `!curl https://sdk.cloud.google.com`), breaking cross-platform portability on Windows and macOS.
3. **Absence of Preprocessing & Inference Modules**:
   - No standalone inference script was provided to run flood segmentation on a single chip or newly acquired GeoTIFF.
4. **Missing Dependency Manifest**:
   - No `requirements.txt` or `pyproject.toml` existed in the repository.
5. **No Front-End or Visual Tooling**:
   - Users had no interface to inspect multi-spectral composites, visualize radar backscatter, test thresholding algorithms interactively, or view spatial flood metrics.

---

## 4. Modernization & Front-End Implementation

To resolve all identified gaps, the repository has been upgraded with:

### A. Core Python Package (`sen1floods11/`)
- `models.py`: Defines the adapted `FCN-ResNet50` architecture with modified 2-channel `conv1` and recursive `convert_bn_to_gn` GroupNorm conversion. Includes checkpoint weight loader.
- `preprocessing.py`: Implements Sen1Floods11 dB clipping ($[-50, 1]\text{ dB} \to [0, 1]$), channel-wise Z-score normalization ($\mu = [0.6851, 0.5235], \sigma = [0.0820, 0.1102]$), and optical spectral index computation (NDWI, MNDWI).
- `inference.py`: Implements four flood detection algorithms:
  1. **Adaptive Otsu Thresholding** (pure NumPy vectorized between-class variance maximization on VH radar).
  2. **Calibrated Metadata VH Threshold** (country-specific dB threshold).
  3. **Optical NDWI / MNDWI Thresholding** (Sentinel-2 green vs. NIR/SWIR).
  4. **Deep Learning Inference** (`FCN-ResNet50` PyTorch forward pass).
- `metrics.py`: Computes Water IoU (Jaccard Index), Overall Accuracy, Precision, Recall, F1-Score, Confusion Matrix (TP, FP, TN, FN, Ignored), and flood area in Hectares and Square Kilometers ($1\text{ pixel} = 100\text{ m}^2 = 0.01\text{ ha} = 0.0001\text{ km}^2$).
- `visualizer.py`: High-contrast percentile stretching ($2\% - 98\%$) for True Color RGB, False Color Infrared (CIR), SWIR, Dual-Pol SAR false color, and classification error heatmaps.

### B. Web Front-End & REST API (`app.py`, `templates/`, `static/`)
- **Single Page Application (SPA)** built with HTML5, CSS3, vanilla JavaScript, Leaflet.js, and Flask.
- **Global Flood Footprint Map**: Visualizes the 11 global flood event polygons, acquisition dates, orbits, and calibrated VH thresholds.
- **Multi-Sensor Band Selector**: Instant switching between Sentinel-2 RGB, CIR, SWIR, NDWI, Sentinel-1 Dual-Pol, VV, VH, and Ground Truth.
- **Interactive Segmentation Studio**: Real-time segmentation with adjustable sliders, opacity overlay, side-by-side comparison, and error breakdown.
- **Custom Upload & Export**: Drag-and-drop GeoTIFF upload and one-click mask export.

---

## 5. Quickstart & Verification

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Web Application
python app.py

# 3. Open in browser
http://127.0.0.1:5000
```
