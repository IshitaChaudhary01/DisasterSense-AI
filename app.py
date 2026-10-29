import os
import json
import time
import io
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import tifffile
from PIL import Image

import torch
import sen1floods11 as sf
import src.geo_utils as geo_utils
import src.report_generator as report_gen
from src.ssl_models import SAR_UNet
from src.dataset import normalize_s1

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample")
DATASET_HAND_DIR = os.path.join(BASE_DIR, "datasets", "Sen1Floods11", "v1.1", "data", "flood_events", "HandLabeled")
GEOJSON_PATH = os.path.join(BASE_DIR, "Sen1Floods11_Metadata.geojson")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Determine PyTorch device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache loaded sample files, models, and metadata
CACHE = {
    "samples": {},
    "fcn_model": None,
    "ssl_model": None,
    "supervised_model": None,
    "events": None
}


def get_or_load_fcn_model():
    if CACHE["fcn_model"] is None:
        try:
            CACHE["fcn_model"] = sf.build_fcn_resnet50(in_channels=2, num_classes=2, pretrained_backbone=False)
            CACHE["fcn_model"].to(DEVICE)
            CACHE["fcn_model"].eval()
        except Exception as e:
            print(f"Error initializing FCN model: {e}")
    return CACHE["fcn_model"]


def get_or_load_ssl_unet():
    if CACHE["ssl_model"] is None:
        ckpt = os.path.join(CHECKPOINT_DIR, "ssl_finetuned_model.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Model checkpoint not found: '{ckpt}'. "
                "The Self-Supervised Fine-Tuned checkpoint must exist to run inference."
            )
        try:
            model = SAR_UNet(in_channels=2, num_classes=2, backbone="resnet18")
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            CACHE["ssl_model"] = model
            print(f"Loaded SSL Fine-Tuned checkpoint from {ckpt}")
        except Exception as e:
            raise RuntimeError(f"Failed to load SSL Fine-Tuned model from '{ckpt}': {str(e)}")
    return CACHE["ssl_model"]


def get_or_load_supervised_unet():
    if CACHE["supervised_model"] is None:
        ckpt = os.path.join(CHECKPOINT_DIR, "supervised_baseline.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Model checkpoint not found: '{ckpt}'. "
                "The Supervised Baseline checkpoint must exist to run inference."
            )
        try:
            model = SAR_UNet(in_channels=2, num_classes=2, backbone="resnet18")
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            CACHE["supervised_model"] = model
            print(f"Loaded Supervised Baseline checkpoint from {ckpt}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Supervised Baseline model from '{ckpt}': {str(e)}")
    return CACHE["supervised_model"]


def load_events_metadata():
    if CACHE["events"] is None and os.path.exists(GEOJSON_PATH):
        try:
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                CACHE["events"] = data
        except Exception as e:
            print(f"Error loading GeoJSON metadata: {e}")
            CACHE["events"] = {"features": []}
    return CACHE["events"]


def scan_sample_chips():
    samples = []
    seen_ids = set()

    # 1. Local sample folder
    s1_sample_dir = os.path.join(SAMPLE_DIR, "S1")
    if os.path.exists(s1_sample_dir):
        for f in os.listdir(s1_sample_dir):
            if f.endswith(".tif"):
                chip_id = f.replace("_S1Hand.tif", "").replace(".tif", "")
                parts = chip_id.split("_")
                country = parts[0] if len(parts) > 0 else "Unknown"
                num_id = parts[1] if len(parts) > 1 else ""
                s2_file = os.path.join(SAMPLE_DIR, "S2", f"{chip_id}_S2Hand.tif")
                label_file = os.path.join(SAMPLE_DIR, "Labels", f"{chip_id}_LabelHand.tif")
                
                samples.append({
                    "id": chip_id,
                    "country": country,
                    "chip_num": num_id,
                    "has_s1": True,
                    "has_s2": os.path.exists(s2_file),
                    "has_label": os.path.exists(label_file),
                    "s1_path": os.path.join(s1_sample_dir, f),
                    "s2_path": s2_file if os.path.exists(s2_file) else None,
                    "label_path": label_file if os.path.exists(label_file) else None
                })
                seen_ids.add(chip_id)

    # 2. Downloaded dataset folder
    dataset_s1_dir = os.path.join(DATASET_HAND_DIR, "S1Hand")
    dataset_s2_dir = os.path.join(DATASET_HAND_DIR, "S2Hand")
    dataset_label_dir = os.path.join(DATASET_HAND_DIR, "LabelHand")

    if os.path.exists(dataset_s1_dir):
        for f in os.listdir(dataset_s1_dir):
            if f.endswith(".tif"):
                chip_id = f.replace("_S1Hand.tif", "").replace(".tif", "")
                if chip_id in seen_ids:
                    continue
                parts = chip_id.split("_")
                country = parts[0] if len(parts) > 0 else "Unknown"
                num_id = parts[1] if len(parts) > 1 else ""
                s2_file = os.path.join(dataset_s2_dir, f"{chip_id}_S2Hand.tif")
                label_file = os.path.join(dataset_label_dir, f"{chip_id}_LabelHand.tif")

                samples.append({
                    "id": chip_id,
                    "country": country,
                    "chip_num": num_id,
                    "has_s1": True,
                    "has_s2": os.path.exists(s2_file),
                    "has_label": os.path.exists(label_file),
                    "s1_path": os.path.join(dataset_s1_dir, f),
                    "s2_path": s2_file if os.path.exists(s2_file) else None,
                    "label_path": label_file if os.path.exists(label_file) else None
                })
                seen_ids.add(chip_id)

    return samples


def get_chip_data(chip_id):
    if chip_id in CACHE["samples"]:
        return CACHE["samples"][chip_id]

    samples = scan_sample_chips()
    target = next((s for s in samples if s["id"] == chip_id), None)

    if not target:
        # Check uploads
        upload_meta_path = os.path.join(UPLOAD_DIR, f"{chip_id}_meta.json")
        if os.path.exists(upload_meta_path):
            with open(upload_meta_path, "r", encoding="utf-8") as f:
                target = json.load(f)
        else:
            return None

    data = {"info": target}
    if target.get("s1_path") and os.path.exists(target["s1_path"]):
        data["s1"] = tifffile.imread(target["s1_path"])
        data["geo_info"] = geo_utils.inspect_image_file(target["s1_path"])
    elif target.get("image_path") and os.path.exists(target["image_path"]):
        data["geo_info"] = geo_utils.inspect_image_file(target["image_path"])

    if target.get("s2_path") and os.path.exists(target["s2_path"]):
        data["s2"] = tifffile.imread(target["s2_path"])
    if target.get("label_path") and os.path.exists(target["label_path"]):
        data["label"] = tifffile.imread(target["label_path"])

    CACHE["samples"][chip_id] = data
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    events = load_events_metadata()
    samples = scan_sample_chips()
    ssl_ready = os.path.exists(os.path.join(CHECKPOINT_DIR, "ssl_finetuned_model.pt")) or os.path.exists(os.path.join(CHECKPOINT_DIR, "ssl_pretrained_encoder.pt"))
    sup_ready = os.path.exists(os.path.join(CHECKPOINT_DIR, "supervised_baseline.pt"))

    return jsonify({
        "status": "ready",
        "project_title": "Self-Supervised Deep Learning for Remote Sensing-Based Disaster Detection",
        "device": DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "events_count": len(events.get("features", [])) if events else 0,
        "sample_chips_count": len(samples),
        "dataset_name": "Sen1Floods11",
        "resolution": "10 meters / pixel (WGS 84 / EPSG:4326)",
        "models_available": {
            "ssl_simclr_unet": ssl_ready,
            "supervised_unet": sup_ready,
            "fcn_resnet50": True,
            "otsu_baseline": True,
            "ndwi_optical": True
        }
    })


@app.route("/api/events")
def api_events():
    events = load_events_metadata()
    return jsonify(events)


@app.route("/api/samples")
def api_samples():
    samples = scan_sample_chips()
    events = load_events_metadata()
    country_meta = {}
    if events and "features" in events:
        for f in events["features"]:
            props = f.get("properties", {})
            loc = props.get("location")
            if loc:
                country_meta[loc.lower()] = props

    enriched = []
    for s in samples:
        c_lower = s["country"].lower()
        m = country_meta.get(c_lower, {})
        s["vh_thresh"] = m.get("VH_thresh", -22.0)
        s["s1_date"] = m.get("s1_date", "N/A")
        s["s2_date"] = m.get("s2_date", "N/A")
        s["orbit"] = m.get("orbit", "N/A")
        enriched.append(s)

    return jsonify(enriched)


@app.route("/api/sample/<chip_id>/visualize")
def api_visualize_sample(chip_id):
    data = get_chip_data(chip_id)
    if not data:
        return jsonify({"error": f"Chip '{chip_id}' not found"}), 404

    result = {
        "chip_id": chip_id,
        "info": data["info"],
        "geo_info": data.get("geo_info", {}),
        "views": {}
    }

    # Sentinel-1 views
    if "s1" in data:
        s1 = data["s1"]
        s1_vis = sf.create_s1_composite(s1)
        result["views"]["s1_false_color"] = sf.array_to_base64_png(s1_vis["false_color"])
        result["views"]["s1_vv"] = sf.array_to_base64_png(s1_vis["vv"])
        result["views"]["s1_vh"] = sf.array_to_base64_png(s1_vis["vh"])
        result["dimensions"] = [int(s1.shape[1]), int(s1.shape[2])]
        result["s1_min_max"] = {
            "vv_min": float(round(s1[0].min(), 2)),
            "vv_max": float(round(s1[0].max(), 2)),
            "vh_min": float(round(s1[1].min(), 2)),
            "vh_max": float(round(s1[1].max(), 2))
        }

    # Sentinel-2 views
    if "s2" in data:
        s2 = data["s2"]
        s2_vis = sf.create_s2_composite(s2)
        result["views"]["s2_rgb"] = sf.array_to_base64_png(s2_vis["true_color"])
        result["views"]["s2_cir"] = sf.array_to_base64_png(s2_vis["false_color_cir"])
        result["views"]["s2_swir"] = sf.array_to_base64_png(s2_vis["swir"])

        # NDWI Preview map
        indices = sf.preprocessing.compute_spectral_indices(s2)
        ndwi_norm = ((np.clip(indices["ndwi"], -0.5, 0.5) + 0.5) * 255.0).astype(np.uint8)
        ndwi_rgb = np.stack([
            255 - ndwi_norm,
            ndwi_norm,
            np.clip(ndwi_norm * 1.5, 0, 255).astype(np.uint8)
        ], axis=-1)
        result["views"]["s2_ndwi"] = sf.array_to_base64_png(ndwi_rgb)

    # Custom upload RGB view if available
    if "rgb_image" in data:
        result["views"]["custom_rgb"] = sf.array_to_base64_png(data["rgb_image"])

    # Ground truth
    if "label" in data:
        gt_rendered = sf.visualizer.render_ground_truth(data["label"])
        result["views"]["ground_truth"] = sf.array_to_base64_png(gt_rendered)
        label_arr = data["label"]
        water_px = int(np.sum(label_arr == 1))
        land_px = int(np.sum(label_arr == 0))
        invalid_px = int(np.sum((label_arr < 0) | (label_arr == 255)))
        result["ground_truth_stats"] = {
            "water_pixels": water_px,
            "land_pixels": land_px,
            "invalid_pixels": invalid_px,
            "water_percentage": round(water_px / (water_px + land_px) * 100, 2) if (water_px + land_px) > 0 else 0
        }

    return jsonify(result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json() or {}
    chip_id = payload.get("chip_id", "Spain_7370579")
    method = payload.get("method", "ssl_unet").lower()
    custom_threshold = payload.get("threshold", None)
    base_image_type = payload.get("base_image_type", "s2_rgb")
    alpha = float(payload.get("opacity", 0.55))

    try:
        data = get_chip_data(chip_id)
        if not data:
            return jsonify({"error": f"Chip '{chip_id}' not found"}), 404

        s1_data = data.get("s1")
        s2_data = data.get("s2")
        gt_data = data.get("label")
        rgb_data = data.get("rgb_image")
        geo_info = data.get("geo_info", {})

        start_time = time.time()
        water_mask = None
        confidence_map = None
        applied_threshold = None
        mean_confidence = None
        method_name = ""

        # 1. Self-Supervised SimCLR Pretrained U-Net (Primary Project Model)
        if method == "ssl_unet":
            if s1_data is None:
                return jsonify({
                    "error": "Incompatible imagery: Model expects 2-channel Sentinel-1 SAR data (VV, VH in dB). "
                             "This scene does not have SAR channels available. If testing with optical imagery, "
                             "switch to 'RGB Demonstration Mode' or select a benchmark SAR chip."
                }), 400
            model = get_or_load_ssl_unet()
            norm_s1 = normalize_s1(s1_data)
            tensor = torch.from_numpy(norm_s1).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                water_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                water_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy().astype(np.float32)

            mean_confidence = float(water_prob[water_mask == 1].mean()) if (water_mask == 1).sum() > 0 else float(water_prob.mean())
            confidence_map = (np.clip(water_prob, 0, 1) * 255.0).astype(np.uint8)
            method_name = f"Self-Supervised SimCLR Pretrained U-Net ({DEVICE.upper()})"

        # 2. Supervised Baseline U-Net
        elif method == "supervised_unet":
            if s1_data is None:
                return jsonify({
                    "error": "Incompatible imagery: Model expects 2-channel Sentinel-1 SAR data (VV, VH in dB)."
                }), 400
            model = get_or_load_supervised_unet()
            norm_s1 = normalize_s1(s1_data)
            tensor = torch.from_numpy(norm_s1).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                water_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                water_prob = probs[:, 1, :, :].squeeze(0).cpu().numpy().astype(np.float32)

            mean_confidence = float(water_prob[water_mask == 1].mean()) if (water_mask == 1).sum() > 0 else float(water_prob.mean())
            confidence_map = (np.clip(water_prob, 0, 1) * 255.0).astype(np.uint8)
            method_name = f"Supervised Baseline U-Net ({DEVICE.upper()})"

        # 3. FCN-ResNet50 Deep Learning Model
        elif method == "fcn":
            if s1_data is None:
                return jsonify({"error": "Sentinel-1 data required for FCN Deep Learning model"}), 400
            model = get_or_load_fcn_model()
            water_mask, water_prob = sf.predict_fcn(model, s1_data, device=DEVICE)
            mean_confidence = float(water_prob[water_mask == 1].mean()) if (water_mask == 1).sum() > 0 else float(water_prob.mean())
            confidence_map = (np.clip(water_prob, 0, 1) * 255.0).astype(np.uint8)
            method_name = f"FCN-ResNet50 Deep Learning Model ({DEVICE.upper()})"

        # 4. Adaptive Otsu Thresholding
        elif method == "otsu":
            if s1_data is None:
                return jsonify({"error": "Sentinel-1 SAR data required for Otsu thresholding"}), 400
            water_mask, applied_threshold = sf.predict_otsu(s1_data)
            method_name = f"Adaptive Otsu Thresholding ({applied_threshold:.2f} dB)"

        # 5. Calibrated Metadata VH Threshold
        elif method == "metadata_vh":
            if s1_data is None:
                return jsonify({"error": "Sentinel-1 data required for VH thresholding"}), 400
            th = -22.0
            events = load_events_metadata()
            country = data["info"].get("country", "").lower()
            if events and "features" in events:
                for f in events["features"]:
                    props = f.get("properties", {})
                    if props.get("location", "").lower() == country:
                        th = props.get("VH_thresh", -22.0)
                        break
            water_mask, applied_threshold = sf.predict_vh_threshold(s1_data, threshold_db=th)
            method_name = f"Sen1Floods11 Calibrated VH Threshold ({applied_threshold:.2f} dB)"

        # 6. Optical NDWI / MNDWI
        elif method in ["ndwi", "mndwi"]:
            if s2_data is None:
                return jsonify({"error": "Sentinel-2 optical data required for spectral indices"}), 400
            th = float(custom_threshold) if custom_threshold is not None else 0.0
            water_mask, idx_arr = sf.predict_optical_ndwi(s2_data, index_name=method, threshold=th)
            applied_threshold = th
            method_name = f"Sentinel-2 Optical {method.upper()} (Threshold > {th:.2f})"

        # 7. RGB Demonstration Mode
        elif method == "rgb_demo":
            if rgb_data is None and s2_data is not None:
                s2_vis = sf.create_s2_composite(s2_data)
                rgb_data = s2_vis["true_color"]
            elif rgb_data is None:
                return jsonify({"error": "RGB image data not available"}), 400

            # Normalized Blue Index / Excess Blue: (B - R) / (B + R + 1e-6)
            r = rgb_data[:, :, 0].astype(np.float32)
            b = rgb_data[:, :, 2].astype(np.float32)
            nbi = (b - r) / (b + r + 1e-6)
            water_mask = (nbi > 0.05).astype(np.uint8)
            method_name = "RGB Optical Demonstration Mode [Caution: No Cloud Penetration]"

        else:
            return jsonify({"error": f"Unknown prediction method '{method}'"}), 400

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Determine base image for overlay
        base_rgb = None
        if s2_data is not None and base_image_type in ["s2_rgb", "s2_cir"]:
            s2_composites = sf.create_s2_composite(s2_data)
            base_rgb = s2_composites["true_color"] if base_image_type == "s2_rgb" else s2_composites["false_color_cir"]
        elif s1_data is not None:
            s1_composites = sf.create_s1_composite(s1_data)
            if base_image_type == "s1_vv":
                base_rgb = s1_composites["vv"]
            elif base_image_type == "s1_vh":
                base_rgb = s1_composites["vh"]
            else:
                base_rgb = s1_composites["false_color"]
        elif rgb_data is not None:
            base_rgb = rgb_data
        else:
            base_rgb = np.zeros((water_mask.shape[0], water_mask.shape[1], 3), dtype=np.uint8)

        # Render visualizations
        overlay_img = sf.render_mask_overlay(base_rgb, water_mask, color=(0, 220, 255), alpha=alpha)
        pred_mask_vis = sf.visualizer.render_prediction_mask(water_mask)
        binary_vis = np.zeros_like(base_rgb)
        binary_vis[water_mask == 1] = [0, 220, 255]

        # Calculate real-world affected area strictly using validated geospatial metadata
        area_stats = geo_utils.compute_geospatial_affected_area(water_mask, geo_info)

        # Confidence map visualization (cyan heatmap)
        confidence_b64 = None
        if confidence_map is not None:
            conf_rgb = np.stack([
                np.zeros_like(confidence_map),
                (confidence_map * 0.85).astype(np.uint8),
                confidence_map
            ], axis=-1)
            confidence_b64 = sf.array_to_base64_png(conf_rgb)

        # Ground truth evaluation metrics if available
        metrics = None
        error_map_b64 = None
        if gt_data is not None:
            metrics = sf.evaluate_segmentation(water_mask, gt_data)
            error_map_arr = sf.render_error_map(water_mask, gt_data)
            error_map_b64 = sf.array_to_base64_png(error_map_arr)

        # Automated Natural Language Disaster Assessment Report
        report_dict = report_gen.generate_disaster_report({
            "chip_id": chip_id,
            "method_name": method_name,
            "area_stats": area_stats,
            "geo_info": geo_info,
            "confidence": {
                "mean_prob": f"{mean_confidence * 100:.1f}%" if mean_confidence is not None else "N/A",
                "raw_mean": round(mean_confidence, 4) if mean_confidence is not None else None
            },
            "metrics": metrics
        })

        return jsonify({
            "success": True,
            "method": method,
            "method_name": method_name,
            "applied_threshold": applied_threshold,
            "execution_time_ms": elapsed_ms,
            "area_stats": area_stats,
            "metrics": metrics,
            "confidence": {
                "mean_score": round(mean_confidence, 4) if mean_confidence is not None else None,
                "mean_score_pct": f"{mean_confidence * 100:.1f}%" if mean_confidence is not None else "N/A"
            },
            "report": report_dict,
            "views": {
                "overlay": sf.array_to_base64_png(overlay_img),
                "prediction_mask": sf.array_to_base64_png(pred_mask_vis),
                "binary_mask": sf.array_to_base64_png(binary_vis),
                "error_map": error_map_b64,
                "confidence_map": confidence_b64
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference execution failed: {str(e)}"}), 500


@app.route("/api/inspect_upload", methods=["POST"])
def api_inspect_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file attached"}), 400
    file = request.files["file"]
    filename = file.filename
    clean_id = f"inspect_{int(time.time())}"
    temp_path = os.path.join(UPLOAD_DIR, f"{clean_id}_{filename}")
    file.save(temp_path)

    inspection = geo_utils.inspect_image_file(temp_path)
    return jsonify({
        "success": True,
        "file_name": filename,
        "inspection": inspection
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = file.filename
    clean_id = "upload_" + str(int(time.time()))
    save_path = os.path.join(UPLOAD_DIR, f"{clean_id}_{filename}")
    file.save(save_path)

    geo_info = geo_utils.inspect_image_file(save_path)

    # Parse imagery
    try:
        if filename.lower().endswith((".tif", ".tiff")):
            arr = tifffile.imread(save_path)
        else:
            img = Image.open(save_path).convert("RGB")
            arr = np.array(img)
    except Exception as e:
        return jsonify({"error": f"Failed to parse uploaded raster: {str(e)}"}), 400

    info = {
        "id": clean_id,
        "country": "Custom Upload",
        "chip_num": filename,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "geo_info": geo_info
    }

    chip_entry = {"info": info, "geo_info": geo_info}

    # Transpose channels if needed
    if arr.ndim == 3 and arr.shape[2] == 2:
        arr = np.transpose(arr, (2, 0, 1))

    if arr.ndim == 3 and arr.shape[0] == 2:
        # 2-channel Sentinel-1 SAR (VV, VH)
        chip_entry["s1"] = arr
        info["has_s1"] = True
        info["has_s2"] = False
        info["s1_path"] = save_path
    elif arr.ndim == 3 and arr.shape[0] >= 12:
        # Sentinel-2 Multispectral
        chip_entry["s2"] = arr
        info["has_s1"] = False
        info["has_s2"] = True
        info["s2_path"] = save_path
    else:
        # RGB or Grayscale image
        rgb_arr = arr
        if rgb_arr.ndim == 3 and rgb_arr.shape[0] in [3, 4]:
            rgb_arr = np.transpose(rgb_arr[:3], (1, 2, 0))
        elif rgb_arr.ndim == 3 and rgb_arr.shape[2] == 4:
            rgb_arr = rgb_arr[:, :, :3]
        elif rgb_arr.ndim == 2:
            u8 = sf.visualizer.percentile_stretch(rgb_arr)
            rgb_arr = np.stack([u8, u8, u8], axis=-1)

        if rgb_arr.dtype != np.uint8:
            rgb_arr = np.stack([sf.visualizer.percentile_stretch(rgb_arr[:, :, c]) for c in range(3)], axis=-1)

        chip_entry["rgb_image"] = rgb_arr
        info["has_s1"] = False
        info["has_s2"] = False
        info["image_path"] = save_path

    info["has_label"] = False
    with open(os.path.join(UPLOAD_DIR, f"{clean_id}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(info, f)

    CACHE["samples"][clean_id] = chip_entry

    return jsonify({
        "success": True,
        "chip_id": clean_id,
        "info": info,
        "inspection": geo_info,
        "geo_info": geo_info
    })


@app.route("/api/benchmark_metrics")
def api_benchmark_metrics():
    metrics_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    chart_path = os.path.join(RESULTS_DIR, "benchmark_comparison.png")

    data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    chart_b64 = None
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            import base64
            chart_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

    return jsonify({
        "metrics": data,
        "chart_b64": chart_b64
    })


@app.route("/api/download_report", methods=["POST"])
def api_download_report():
    payload = request.get_json() or {}
    report_text = payload.get("markdown", "# Remote Sensing Disaster Report\nNo report data available.")
    buf = io.BytesIO(report_text.encode("utf-8"))
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="flood_disaster_assessment_report.md",
        mimetype="text/markdown"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Sen1Floods11 Disaster Studio on port {port}...")
    app.run(host="127.0.0.1", port=port, debug=False)
