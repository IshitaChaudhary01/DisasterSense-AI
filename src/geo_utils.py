import os
import math
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def inspect_image_file(file_path):
    """
    Inspects and validates an uploaded satellite image or standard image file.
    Extracts geospatial metadata, CRS, resolution, channel information, and suitability for SAR AI model.
    """
    ext = os.path.splitext(file_path)[1].lower()
    file_size_kb = os.path.getsize(file_path) / 1024.0

    info = {
        "file_name": os.path.basename(file_path),
        "file_size_kb": round(file_size_kb, 2),
        "extension": ext,
        "is_geospatial": False,
        "crs": None,
        "pixel_size_x_m": None,
        "pixel_size_y_m": None,
        "pixel_area_m2": None,
        "channels": 0,
        "dtype": "unknown",
        "dimensions": [0, 0],
        "is_sar_compatible": False,
        "is_rgb_demo": False,
        "compatibility_notes": ""
    }

    # 1. GeoTIFF Inspection
    if ext in [".tif", ".tiff"] and HAS_RASTERIO:
        try:
            with rasterio.open(file_path) as src:
                info["is_geospatial"] = bool(src.crs is not None)
                info["crs"] = str(src.crs) if src.crs else "Unreferenced"
                info["channels"] = src.count
                info["dimensions"] = [src.height, src.width]
                info["dtype"] = str(src.dtypes[0])

                # Calculate ground resolution in meters
                if src.crs:
                    bounds = src.bounds
                    mid_lat = (bounds.bottom + bounds.top) / 2.0
                    res_x, res_y = src.res

                    if src.crs.is_geographic:
                        # EPSG:4326 (degrees) -> convert to meters
                        m_per_deg_lat = 111320.0
                        m_per_deg_lon = 111320.0 * math.cos(math.radians(mid_lat))
                        px_m = abs(res_x) * m_per_deg_lon
                        py_m = abs(res_y) * m_per_deg_lat
                    else:
                        # Projected CRS (UTM etc. in meters)
                        px_m = abs(res_x)
                        py_m = abs(res_y)

                    info["pixel_size_x_m"] = round(px_m, 2)
                    info["pixel_size_y_m"] = round(py_m, 2)
                    info["pixel_area_m2"] = round(px_m * py_m, 2)

                # SAR Compatibility check
                if src.count == 2 and "float" in info["dtype"].lower():
                    info["is_sar_compatible"] = True
                    info["compatibility_notes"] = "Valid Sentinel-1 Dual-Pol SAR (VV + VH in dB). Fully compatible with AI model."
                elif src.count == 2:
                    info["is_sar_compatible"] = True
                    info["compatibility_notes"] = "2-channel GeoTIFF. Interpreted as VV/VH SAR bands."
                elif src.count >= 12:
                    info["compatibility_notes"] = f"Multispectral Optical GeoTIFF ({src.count} bands, e.g. Sentinel-2). Suitable for NDWI/MNDWI analysis."
                elif src.count in [3, 4]:
                    info["is_rgb_demo"] = True
                    info["compatibility_notes"] = "Georeferenced RGB GeoTIFF. Analyzable in RGB Demonstration Mode."
                else:
                    info["compatibility_notes"] = f"Single-channel or {src.count}-channel GeoTIFF."
                return info
        except Exception as e:
            info["compatibility_notes"] = f"Failed to read GeoTIFF headers: {str(e)}"

    # 2. Standard Image (PNG, JPG, JPEG)
    try:
        with Image.open(file_path) as img:
            info["dimensions"] = [img.height, img.width]
            info["channels"] = len(img.getbands())
            info["dtype"] = img.mode
            info["is_geospatial"] = False
            info["is_sar_compatible"] = False
            info["is_rgb_demo"] = True
            info["compatibility_notes"] = (
                "Standard RGB/Grayscale image without geospatial metadata or SAR radar bands. "
                "The trained AI flood model expects 2-channel Sentinel-1 SAR (VV + VH in dB). "
                "Processed in Demonstration Mode only; real-world area cannot be estimated."
            )
            return info
    except Exception as e:
        info["compatibility_notes"] = f"Unrecognized image format: {str(e)}"
        return info


def compute_geospatial_affected_area(water_mask, geo_info):
    """
    Computes real-world affected surface area strictly when geospatial metadata is verified.
    Accounts for invalid/no-data pixels (-1 or 255).
    """
    mask = np.asarray(water_mask, dtype=np.int32)
    valid_pixels = int(np.sum((mask >= 0) & (mask != 255)))
    water_pixels = int(np.sum(mask == 1))

    pct = (water_pixels / valid_pixels * 100.0) if valid_pixels > 0 else 0.0

    if not geo_info.get("is_geospatial") or not geo_info.get("pixel_area_m2"):
        return {
            "can_calculate_real_area": False,
            "reason": "Image lacks georeferenced Coordinate Reference System (CRS) or ground resolution. Real-world area cannot be scientifically estimated.",
            "water_pixels": water_pixels,
            "valid_pixels": valid_pixels,
            "water_percentage": round(pct, 2),
            "area_m2": None,
            "area_hectares": None,
            "area_sq_km": None
        }

    px_area_m2 = geo_info["pixel_area_m2"]
    total_area_m2 = water_pixels * px_area_m2
    hectares = total_area_m2 / 10000.0
    sq_km = total_area_m2 / 1000000.0

    return {
        "can_calculate_real_area": True,
        "crs": geo_info["crs"],
        "pixel_resolution_m": f"{geo_info['pixel_size_x_m']}m x {geo_info['pixel_size_y_m']}m",
        "water_pixels": water_pixels,
        "valid_pixels": valid_pixels,
        "water_percentage": round(pct, 2),
        "area_m2": round(total_area_m2, 2),
        "area_hectares": round(hectares, 2),
        "area_sq_km": round(sq_km, 3)
    }
