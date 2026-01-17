import json

# from datetime import datetime
import folium
import h3
import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
from folium import plugins
from matplotlib.colors import LinearSegmentedColormap

from .models import AnalysisMetadata, SummaryStatistics

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------


def _normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has 'lat' and 'lon' columns.
    Supports backward compatibility.
    """
    if "lat" in df.columns and "lon" in df.columns:
        return df

    if "center_lat" in df.columns and "center_lon" in df.columns:
        df = df.rename(columns={"center_lat": "lat", "center_lon": "lon"})
        return df

    raise ValueError(
        "DataFrame must contain either "
        + "['lat', 'lon'] or ['center_lat', 'center_lon'] columns.\n"
        + f"Found columns: {list(df.columns)}"
    )


# -----------------------------------------------------------------------------
# COLORMAPS
# -----------------------------------------------------------------------------


def create_change_colormap():
    colors = [
        "#8B0000",
        "#FF4500",
        "#FFA500",
        "#F0F0F0",
        "#90EE90",
        "#228B22",
        "#006400",
    ]
    return LinearSegmentedColormap.from_list("ndvi_change", colors, N=256)


def create_ndvi_colormap():
    return LinearSegmentedColormap.from_list(
        "ndvi",
        [
            "#8B0000",
            "#FF4500",
            "#FFA500",
            "#F0F0F0",
            "#90EE90",
            "#228B22",
            "#006400",
        ],
        N=256,
    )


# -----------------------------------------------------------------------------
# SUMMARY STATISTICS
# -----------------------------------------------------------------------------


def generate_summary_statistics(
    df: pd.DataFrame, h3_resolution: int = 9
) -> SummaryStatistics:
    """
    Generate comprehensive summary statistics from results DataFrame.
    """
    df = _normalize_coordinates(df)

    # Basic counts
    total = len(df)
    vegetated = df[df["is_vegetated"]].copy()
    veg_count = len(vegetated)
    non_veg_count = total - veg_count

    # Change statistics (vegetated only)
    if veg_count > 0:
        mean_change = vegetated["change"].mean()
        median_change = vegetated["change"].median()
        std_change = vegetated["change"].std()
        min_change = vegetated["change"].min()
        max_change = vegetated["change"].max()

        # Direction counts
        increase = len(vegetated[vegetated["direction"] == "Increase"])
        decrease = len(vegetated[vegetated["direction"] == "Decrease"])
        stable = len(vegetated[vegetated["direction"] == "Stable"])

        # Severity counts
        severe = len(vegetated[vegetated["severity"] == "Severe"])
        moderate = len(vegetated[vegetated["severity"] == "Moderate"])
        mild = len(vegetated[vegetated["severity"] == "Mild"])
    else:
        mean_change = median_change = std_change = min_change = max_change = 0.0
        increase = decrease = stable = severe = moderate = mild = 0

    # Area calculations (H3 resolution 9 ≈ 0.1053 km²)
    hex_area_km2 = h3.cell_area(h3.latlng_to_cell(0, 0, h3_resolution), unit="km^2")
    total_area = total * hex_area_km2
    veg_area = veg_count * hex_area_km2
    severe_area = severe * hex_area_km2

    # Quality metrics
    avg_confidence = df["confidence"].mean() if "confidence" in df.columns else 0.0
    high_uncertainty = (
        len(df[df["uncertainty_flag"]]) if "uncertainty_flag" in df.columns else 0
    )

    return SummaryStatistics(
        total_hexagons=total,
        vegetated_hexagons=veg_count,
        non_vegetated_hexagons=non_veg_count,
        mean_change=mean_change,
        median_change=median_change,
        std_change=std_change,
        min_change=min_change,
        max_change=max_change,
        increase_count=increase,
        decrease_count=decrease,
        stable_count=stable,
        severe_count=severe,
        moderate_count=moderate,
        mild_count=mild,
        total_area_km2=total_area,
        vegetated_area_km2=veg_area,
        severe_change_area_km2=severe_area,
        avg_confidence=avg_confidence,
        high_uncertainty_count=high_uncertainty,
    )


# -----------------------------------------------------------------------------
# STATIC MATPLOTLIB VISUALIZATION
# -----------------------------------------------------------------------------


def create_matplotlib_visualization(df: pd.DataFrame, output_file: str):
    """Enhanced static visualization with confidence overlay."""
    print("Creating static visualization...")

    df = _normalize_coordinates(df)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("NDVI Analysis – Enhanced Report", fontsize=16, fontweight="bold")

    # Plot 1: NDVI 2025
    sc1 = axes[0, 0].scatter(
        df["lon"],
        df["lat"],
        c=df["ndvi_2025"],
        cmap="YlGn",
        s=40,
        alpha=0.7,
    )
    axes[0, 0].set_title("NDVI – Jan 2025")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")
    plt.colorbar(sc1, ax=axes[0, 0])

    # Plot 2: NDVI 2026
    sc2 = axes[0, 1].scatter(
        df["lon"],
        df["lat"],
        c=df["ndvi_2026"],
        cmap="YlGn",
        s=40,
        alpha=0.7,
    )
    axes[0, 1].set_title("NDVI – Jan 2026")
    axes[0, 1].set_xlabel("Longitude")
    plt.colorbar(sc2, ax=axes[0, 1])

    # Plot 3: Confidence Map
    if "confidence" in df.columns:
        sc3 = axes[0, 2].scatter(
            df["lon"],
            df["lat"],
            c=df["confidence"],
            cmap="RdYlGn",
            s=40,
            alpha=0.7,
            vmin=0,
            vmax=1,
        )
        axes[0, 2].set_title("Confidence Score")
        axes[0, 2].set_xlabel("Longitude")
        plt.colorbar(sc3, ax=axes[0, 2], label="Confidence (0-1)")

    veg = df[df["is_vegetated"]].copy()
    cmap = create_change_colormap()

    if not veg.empty:
        vmax = max(abs(veg["change"].min()), abs(veg["change"].max()))

        # Plot 4: NDVI Change
        sc4 = axes[1, 0].scatter(
            veg["lon"],
            veg["lat"],
            c=veg["change"],
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            s=40,
            alpha=0.7,
        )
        axes[1, 0].set_title("NDVI Change (Vegetated)")
        axes[1, 0].set_xlabel("Longitude")
        axes[1, 0].set_ylabel("Latitude")
        plt.colorbar(sc4, ax=axes[1, 0])

        # Plot 5: Change Distribution
        axes[1, 1].hist(veg["change"], bins=50, color="steelblue", edgecolor="black")
        axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=2)
        axes[1, 1].set_title("Change Distribution")
        axes[1, 1].set_xlabel("NDVI Change")
        axes[1, 1].set_ylabel("Frequency")

        # Plot 6: Severity Breakdown
        if "severity" in veg.columns:
            severity_counts = veg["severity"].value_counts()
            colors_sev = {"Severe": "#DC3545", "Moderate": "#FFC107", "Mild": "#28A745"}
            severity_colors = [
                colors_sev.get(s, "#6C757D") for s in severity_counts.index
            ]

            axes[1, 2].bar(
                severity_counts.index, severity_counts.values, color=severity_colors
            )
            axes[1, 2].set_title("Severity Distribution")
            axes[1, 2].set_xlabel("Severity")
            axes[1, 2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {output_file}")


# -----------------------------------------------------------------------------
# INTERACTIVE MAP (ENHANCED)
# -----------------------------------------------------------------------------


def create_folium_map(df, output_file="ndvi_change_map.html"):
    """Create enhanced interactive map with new fields."""
    print("\nCreating interactive map...")

    df = _normalize_coordinates(df)

    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    )

    # Add tile layers
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        overlay=False,
    ).add_to(m)

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(m)
    folium.TileLayer("Cartodb Positron", name="Light Map", overlay=False).add_to(m)

    # Color mapping
    veg_df = df[df["is_vegetated"]].copy()
    vmax = (
        max(abs(veg_df["change"].min()), abs(veg_df["change"].max()))
        if len(veg_df) > 0
        else 0.3
    )
    cmap = create_change_colormap()

    severity_colors = {
        "Severe": "#DC3545",
        "Moderate": "#FFC107",
        "Mild": "#28A745",
    }

    # Area type colors for non-vegetated
    area_type_colors = {
        "bare_soil_water": "#808080",
        "new_vegetation": "#00BFFF",
        "vegetation_loss": "#FF6347",
    }

    # Add hexagons
    for _, row in df.iterrows():
        boundary = h3.cell_to_boundary(row["hex_id"])
        coords = [[lat, lon] for lat, lon in boundary]

        # Determine color based on area type
        if not row["is_vegetated"]:
            area_type = row.get("area_type", "bare_soil_water")
            hex_color = area_type_colors.get(area_type, "#808080")
            badge_color = "#6C757D"
        else:
            normalized_change = (row["change"] + vmax) / (2 * vmax)
            rgba = cmap(normalized_change)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
            badge_color = severity_colors.get(row.get("severity", "Mild"), "#6C757D")

        # Confidence-based opacity
        confidence = row.get("confidence", 0.7)
        opacity = 0.4 + (confidence * 0.4)  # Range: 0.4-0.8

        # Enhanced popup with new fields
        area_type_display = row.get("area_type", "unknown").replace("_", " ").title()
        confidence_pct = confidence * 100
        anomaly = row.get("anomaly_score", 0)
        uncertainty = row.get("uncertainty_flag", False)

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; min-width: 320px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 10px; margin: -10px -10px 10px -10px;">
                <b style="font-size: 14px;">🌍 Hexagon Analysis</b>
            </div>
            
            <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6;"><b>Hex ID:</b></td>
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6; font-family: monospace;">
                        {row["hex_id"][:12]}...
                    </td>
                </tr>
                <tr>
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6;"><b>Location:</b></td>
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6;">
                        {row["lat"]:.4f}°, {row["lon"]:.4f}°
                    </td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6;"><b>Area Type:</b></td>
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6;">
                        {area_type_display}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6;"><b>Pixels:</b></td>
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6;">
                        {row["pixel_count_2025"]:.0f} (2025) | {
            row["pixel_count_2026"]:.0f} (2026)
                    </td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6;"><b>Confidence:</b></td>
                    <td style="padding: 6px; border-bottom: 1px solid #dee2e6;">
                        <b style="color: {
            "#28A745"
            if confidence > 0.7
            else "#FFC107"
            if confidence > 0.4
            else "#DC3545"
        }">
                            {confidence_pct:.1f}%
                        </b>
                        {"⚠️ Low" if uncertainty else "✓ Good"}
                    </td>
                </tr>
            </table>
            
            {
            '''
            <div style="margin: 10px 0; padding: 10px; background-color: #f0f0f0; border-radius: 4px; text-align: center; font-weight: bold;">
                Non-Vegetated Area
            </div>
            '''
            if not row["is_vegetated"]
            else f'''
            <div style="margin: 12px 0 8px 0; padding: 8px; background-color: #e7f3ff; border-radius: 4px;">
                <b style="color: #0066cc;">📊 2025 Statistics</b>
                <table style="width: 100%; margin-top: 6px; font-size: 12px;">
                    <tr><td>Mean:</td><td style="text-align: right;"><b>{row['ndvi_2025']:.4f}</b></td></tr>
                    <tr><td>Std:</td><td style="text-align: right;">{row['std_2025']:.4f}</td></tr>
                    <tr><td>Range:</td><td style="text-align: right;">{row['ndvi_2025_min']:.3f} - {row['ndvi_2025_max']:.3f}</td></tr>
                </table>
            </div>
            
            <div style="margin: 12px 0 8px 0; padding: 8px; background-color: #e7ffe7; border-radius: 4px;">
                <b style="color: #009900;">📊 2026 Statistics</b>
                <table style="width: 100%; margin-top: 6px; font-size: 12px;">
                    <tr><td>Mean:</td><td style="text-align: right;"><b>{row['ndvi_2026']:.4f}</b></td></tr>
                    <tr><td>Std:</td><td style="text-align: right;">{row['std_2026']:.4f}</td></tr>
                    <tr><td>Range:</td><td style="text-align: right;">{row['ndvi_2026_min']:.3f} - {row['ndvi_2026_max']:.3f}</td></tr>
                </table>
            </div>
            
            <div style="margin: 12px 0; padding: 10px; background-color: {'#d4edda' if row['change'] > 0 else '#f8d7da'}; 
                        border-left: 4px solid {'#28a745' if row['change'] > 0 else '#dc3545'}; border-radius: 4px;">
                <b style="font-size: 14px;">📈 Change Analysis</b>
                <div style="margin-top: 8px;">
                    <div style="margin: 5px 0;">
                        <span style="color: #495057;">Absolute:</span> 
                        <b style="color: {'#28a745' if row['change'] > 0 else '#dc3545'}; font-size: 15px;">
                            {row['change']:+.4f}
                        </b>
                    </div>
                    <div style="margin: 5px 0;">
                        <span style="color: #495057;">Relative:</span> 
                        <b style="color: {'#28a745' if row['change'] > 0 else '#dc3545'}; font-size: 15px;">
                            {row['relative_change']:+.1f}%
                        </b>
                    </div>
                    <div style="margin: 5px 0;">
                        <span style="color: #495057;">Anomaly Score:</span> 
                        <b>{anomaly:+.2f}σ</b>
                    </div>
                </div>
            </div>
            
            <div style="display: flex; gap: 8px; margin-top: 10px; justify-content: space-between;">
                <div style="flex: 1; text-align: center; padding: 8px; 
                            background-color: {badge_color}; color: white; 
                            border-radius: 4px; font-weight: bold; font-size: 11px;">
                    {row.get("severity", "N/A").upper()}
                </div>
                <div style="flex: 1; text-align: center; padding: 8px; 
                            background-color: {'#28a745' if row.get("direction") == "Increase" else '#dc3545' if row.get("direction") == "Decrease" else '#6c757d'}; 
                            color: white; border-radius: 4px; font-weight: bold; font-size: 11px;">
                    {row.get("direction", "N/A").upper()}
                </div>
            </div>
            '''
        }
        </div>
        """

        folium.Polygon(
            locations=coords,
            color="black",
            weight=0.5,
            fill=True,
            fillColor=hex_color,
            fillOpacity=opacity,
            popup=folium.Popup(popup_html, max_width=360),
        ).add_to(m)

    # Enhanced legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 240px; 
                background-color: white; border: 2px solid #495057; 
                border-radius: 8px; z-index: 9999; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                font-size: 13px; padding: 15px; font-family: Arial, sans-serif;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 8px; margin: -15px -15px 12px -15px; 
                    border-radius: 6px 6px 0 0; font-weight: bold;">
            📊 NDVI Change Legend
        </div>
        <p style="margin: 0 0 8px 0; font-size: 11px; color: #6c757d;">
            Jan 2026 vs Jan 2025
        </p>
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="background: linear-gradient(to top, 
                        #006400, #228B22, #90EE90, #F0F0F0, 
                        #FFFF00, #FFA500, #FF4500, #8B0000); 
                        height: 140px; width: 35px; border: 1px solid #ddd; border-radius: 4px;"></div>
            <div style="margin-left: 12px; line-height: 1.8;">
                <div style="margin: 0; font-weight: bold; color: #006400;">+High</div>
                <div style="margin: 48px 0 0 0; font-weight: bold; color: #6c757d;">0</div>
                <div style="margin: 48px 0 0 0; font-weight: bold; color: #8B0000;">-High</div>
            </div>
        </div>
        <div style="border-top: 1px solid #dee2e6; padding-top: 10px; margin-bottom: 10px;">
            <div style="font-weight: bold; margin-bottom: 6px; font-size: 12px;">Severity Levels:</div>
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <div style="width: 12px; height: 12px; background-color: #DC3545; 
                            border-radius: 2px; margin-right: 6px;"></div>
                <span style="font-size: 11px;">Severe</span>
            </div>
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <div style="width: 12px; height: 12px; background-color: #FFC107; 
                            border-radius: 2px; margin-right: 6px;"></div>
                <span style="font-size: 11px;">Moderate</span>
            </div>
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <div style="width: 12px; height: 12px; background-color: #28A745; 
                            border-radius: 2px; margin-right: 6px;"></div>
                <span style="font-size: 11px;">Mild</span>
            </div>
        </div>
        <div style="border-top: 1px solid #dee2e6; padding-top: 10px;">
            <div style="font-weight: bold; margin-bottom: 6px; font-size: 12px;">Area Types:</div>
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <div style="width: 12px; height: 12px; background-color: #00BFFF; 
                            border-radius: 2px; margin-right: 6px;"></div>
                <span style="font-size: 10px;">New Vegetation</span>
            </div>
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <div style="width: 12px; height: 12px; background-color: #FF6347; 
                            border-radius: 2px; margin-right: 6px;"></div>
                <span style="font-size: 10px;">Vegetation Loss</span>
            </div>
            <div style="display: flex; align-items: center; margin: 4px 0;">
                <div style="width: 12px; height: 12px; background-color: #808080; 
                            border-radius: 2px; margin-right: 6px;"></div>
                <span style="font-size: 10px;">Bare/Water</span>
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)

    m.save(output_file)
    print(f"    ✓ Saved: {output_file}")


# -----------------------------------------------------------------------------
# CSV EXPORT WITH VALIDATION
# -----------------------------------------------------------------------------


def save_csv(results, output_path):
    """
    Convert HexResult objects to DataFrame and save to CSV.
    Includes validation warnings.
    """
    if not results:
        raise ValueError("No hex results generated – cannot save CSV.")

    df = pd.DataFrame([r.__dict__ for r in results])

    if df.empty:
        raise ValueError("Resulting DataFrame is empty.")

    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")

    return df


def save_metadata(metadata: AnalysisMetadata, output_path: str):
    """Save analysis metadata as JSON."""
    with open(output_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    print(f"  ✓ Saved metadata: {output_path}")


def save_summary(summary: SummaryStatistics, output_path: str):
    """Save summary statistics as JSON."""
    with open(output_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    print(f"  ✓ Saved summary: {output_path}")
