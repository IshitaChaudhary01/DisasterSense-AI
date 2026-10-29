import datetime

def generate_disaster_report(analysis_data):
    """
    Generates an automated, understandable Disaster Assessment Report.
    Differentiates detected surface water from disaster flooding and includes permanent water context.
    """
    chip_id = analysis_data.get("chip_id", "Unknown Scene")
    method = analysis_data.get("method_name", "AI Flood Segmentation")
    area = analysis_data.get("area_stats", {})
    geo = analysis_data.get("geo_info", {})
    confidence = analysis_data.get("confidence", {})
    metrics = analysis_data.get("metrics")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    water_pct = area.get("water_percentage", 0.0)

    # Automated Natural Language Interpretation
    if water_pct > 35.0:
        severity = "CRITICAL / SEVERE INUNDATION"
        assessment = (
            f"Extensive open water inundation detected across {water_pct}% of the surveyed scene. "
            "High likelihood of catastrophic flood disaster affecting agricultural lowlands and infrastructure. "
            "Immediate humanitarian assessment and civil defense mobilization recommended."
        )
    elif water_pct > 15.0:
        severity = "MODERATE TO HIGH INUNDATION"
        assessment = (
            f"Substantial surface water detected covering {water_pct}% of the scene. "
            "Indicates riverbank overtopping, localized flash flooding, or prolonged agricultural submergence. "
            "Recommended to cross-reference with permanent water baselines (e.g., JRC Global Surface Water)."
        )
    elif water_pct > 3.0:
        severity = "LOCALIZED / MINOR INUNDATION"
        assessment = (
            f"Isolated water bodies detected covering {water_pct}% of the scene. "
            "Likely corresponds to normal river channels, permanent wetlands, or minor localized runoff."
        )
    else:
        severity = "LOW / DRY CONDITIONS"
        assessment = (
            f"Minimal surface water detected ({water_pct}%). "
            "Terrain is predominantly dry with no evidence of major open floodwater inundation."
        )

    # Markdown Report formatting
    md_lines = [
        "# Remote Sensing Disaster Assessment Report",
        f"**Generated:** {now_str}",
        f"**Project:** Self-Supervised Deep Learning for Disaster Detection",
        "",
        "---",
        "## 1. Executive Assessment",
        f"- **Scene Identifier:** `{chip_id}`",
        f"- **Analysis Engine:** {method}",
        f"- **Assessed Disaster Severity:** **{severity}**",
        f"- **Summary:** {assessment}",
        "",
        "---",
        "## 2. Inundation Extent & Surface Measurements"
    ]

    if area.get("can_calculate_real_area", True) and area.get("area_hectares") is not None:
        md_lines.extend([
            f"- **Estimated Flooded Area:** **{area.get('area_hectares', 'N/A')} hectares** ({area.get('area_sq_km', 'N/A')} km^2)",
            f"- **Flooded Pixel Count:** {area.get('water_pixels', 0):,} pixels",
            f"- **Valid Surveyed Pixels:** {area.get('valid_pixels', 0):,} pixels",
            f"- **Proportion of Scene Flooded:** **{water_pct}%**",
            f"- **Spatial Resolution:** {area.get('pixel_resolution_m', '10m x 10m')} (Ground Sampling Distance)",
            f"- **Coordinate Reference System:** `{area.get('crs', 'WGS 84 / EPSG:4326')}`"
        ])
    else:
        md_lines.extend([
            f"- **Flooded Pixel Count:** {area.get('water_pixels', 0):,} pixels ({water_pct}% of scene)",
            f"- **Geospatial Limitation Note:** {area.get('reason', 'Image lacks geospatial projection metadata. Real-world area cannot be estimated.')}"
        ])

    md_lines.extend([
        "",
        "---",
        "## 3. Sensor & Model Characteristics",
        "- **Sensor Modality:** Sentinel-1 C-band Synthetic Aperture Radar (SAR, VV + VH in dB)",
        "- **Cloud Penetration:** Active microwave pulses penetrate storm clouds, rain, and darkness.",
        f"- **Mean Water Detection Confidence:** {confidence.get('mean_prob', 'N/A')}",
        "- **Disaster Water vs. Permanent Water Notice:**",
        "  > *Important:* The model identifies open surface water based on low radar specular backscatter.",
        "  > Permanent water bodies (rivers, lakes, reservoirs) naturally reflect radar similarly to floodwaters.",
        "  > Operators should subtract permanent water baselines (e.g., JRC GSW) to isolate disaster-induced flood extent."
    ])

    if metrics:
        md_lines.extend([
            "",
            "---",
            "## 4. Ground Truth Validation Metrics (Benchmark Chip)",
            f"- **Water IoU (Jaccard Index):** {(metrics.get('iou', 0)*100):.2f}%",
            f"- **Overall Accuracy:** {(metrics.get('accuracy', 0)*100):.2f}%",
            f"- **Precision:** {(metrics.get('precision', 0)*100):.2f}%",
            f"- **Recall:** {(metrics.get('recall', 0)*100):.2f}%",
            f"- **F1-Score / Dice Coefficient:** {(metrics.get('f1', 0)*100):.2f}%"
        ])

    md_lines.extend([
        "",
        "---",
        "## 5. College Project Information",
        "- **Title:** Self-Supervised Deep Learning for Remote Sensing-Based Disaster Detection",
        "- **Dataset:** Sen1Floods11 (Cloud to Street, CVPR Workshops 2020)",
        "- **Pretraining Paradigm:** SimCLR Self-Supervised Contrastive Representation Learning",
        "- **Decoder:** Multi-Scale U-Net with Skip Connections and GroupNorm (16 groups)"
    ])

    full_markdown = "\n".join(md_lines)
    return {
        "markdown": full_markdown,
        "severity": severity,
        "assessment": assessment,
        "timestamp": now_str
    }
