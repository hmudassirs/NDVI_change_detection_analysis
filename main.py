# import json
import os
from datetime import datetime

import yaml

# from tqdm import tqdm
from core.h3_sampling import (
    calculate_spatial_autocorrelation,
    generate_h3_grid,
    sample_hexagons_batch,
    validate_results,
)
from core.io import align_raster, find_band, load_raster
from core.models import AnalysisMetadata
from core.ndvi import compute_ndvi
from core.viz import (
    create_folium_map,
    create_matplotlib_visualization,
    generate_summary_statistics,
    save_csv,
    save_metadata,
    save_summary,
)


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main():
    print("\n" + "=" * 70)
    print("  NDVI CHANGE DETECTION ANALYSIS")
    print("=" * 70 + "\n")

    # Load configuration
    cfg = load_config()
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    # Record start time
    start_time = datetime.now()
    source_files = {}

    # =====================================================================
    # STEP 1: Load Raster Data
    # =====================================================================
    print("📂 STEP 1: Loading Raster Data")
    print("-" * 70)

    print("  → Loading 2025 bands...")
    red25_path = find_band(cfg["paths"]["jan_2025"], cfg["bands"]["red"])
    nir25_path = find_band(cfg["paths"]["jan_2025"], cfg["bands"]["nir"])
    source_files["red_2025"] = red25_path
    source_files["nir_2025"] = nir25_path

    red25, meta25 = load_raster(red25_path)
    nir25, _ = load_raster(nir25_path)
    print(f"    ✓ Loaded 2025 data: {meta25.shape}, CRS: {meta25.crs}")

    print("  → Loading 2026 bands...")
    red26_path = find_band(cfg["paths"]["jan_2026"], cfg["bands"]["red"])
    nir26_path = find_band(cfg["paths"]["jan_2026"], cfg["bands"]["nir"])
    source_files["red_2026"] = red26_path
    source_files["nir_2026"] = nir26_path

    red26, meta26 = load_raster(red26_path)
    nir26, _ = load_raster(nir26_path)
    print(f"    ✓ Loaded 2026 data: {meta26.shape}, CRS: {meta26.crs}")

    # =====================================================================
    # STEP 2: Validate Input Bands
    # =====================================================================
    print("\n🔍 STEP 2: Validating Input Bands")
    print("-" * 70)

    from core.ndvi import diagnose_bands

    # Diagnose 2025 bands
    print("\n  2025 Bands:")
    diagnose_bands(red25, nir25)

    # Diagnose 2026 bands
    print("\n  2026 Bands:")
    diagnose_bands(red26, nir26)

    # =====================================================================
    # STEP 3: Compute NDVI
    # =====================================================================
    print("\n📊 STEP 3: Computing NDVI")
    print("-" * 70)

    print("  → Computing NDVI for 2025...")
    ndvi25 = compute_ndvi(red25, nir25)

    print("\n  → Computing NDVI for 2026...")
    ndvi26 = compute_ndvi(red26, nir26)

    print("\n  → Aligning 2026 to 2025 grid...")
    ndvi26 = align_raster(meta25, ndvi26, meta26)

    # =====================================================================
    # STEP 4: Generate H3 Grid
    # =====================================================================
    print("\n🔷 STEP 4: Generating H3 Hexagonal Grid")
    print("-" * 70)

    hexes = generate_h3_grid(meta25.bounds, meta25.crs, cfg["h3"]["resolution"])
    print(
        f"  ✓ Generated {len(hexes)} hexagons at resolution {cfg['h3']['resolution']}"
    )

    # =====================================================================
    # STEP 5: Sample Hexagons with Progress Bar
    # =====================================================================
    print("\n🔬 STEP 5: Sampling NDVI for Each Hexagon")
    print("-" * 70)

    # Get processing parameters
    n_workers = cfg.get("processing", {}).get("n_workers", 4)
    use_adaptive = cfg.get("processing", {}).get("use_adaptive_severity", True)

    print(f"  → Processing with {n_workers} threads")
    print(f"  → Adaptive severity classification: {use_adaptive}")

    # Combine thresholds
    thresholds = {**cfg["ndvi_thresholds"], **cfg["change_thresholds"]}

    # Process hexagons with batch function (includes progress)
    results = sample_hexagons_batch(
        hex_ids=hexes,
        ndvi_a=ndvi25,
        ndvi_b=ndvi26,
        meta=meta25,
        thresholds=thresholds,
        min_pixel_ratio=cfg["h3"]["min_pixel_ratio"],
        n_workers=n_workers,
        use_adaptive=use_adaptive,
    )

    valid_count = len(results)
    invalid_count = len(hexes) - valid_count

    print("\n  ✓ Sampling complete:")
    print(
        f"    • Valid hexagons: {valid_count}/{len(hexes)} ({valid_count / len(hexes) * 100:.1f}%)"
    )
    print(f"    • Invalid hexagons: {invalid_count}")

    # =====================================================================
    # STEP 6: Validation
    # =====================================================================
    print("\n✅ STEP 6: Validating Results")
    print("-" * 70)

    warnings = validate_results(results)
    if warnings:
        print(f"  ⚠️  Found {len(warnings)} warnings:")
        for i, warning in enumerate(warnings[:5], 1):  # Show first 5
            print(f"    {i}. {warning}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more warnings")
    else:
        print("  ✓ No validation warnings")

    # =====================================================================
    # STEP 7: Spatial Analysis
    # =====================================================================
    print("\n🗺️  STEP 7: Spatial Autocorrelation Analysis")
    print("-" * 70)

    spatial_stats = calculate_spatial_autocorrelation(results)
    if spatial_stats["morans_i"] is not None:
        print(f"  → Moran's I: {spatial_stats['morans_i']:.4f}")
        print(f"  → Interpretation: {spatial_stats['interpretation']}")
    else:
        print("  ⚠️  Insufficient data for spatial analysis")

    # =====================================================================
    # STEP 8: Export Results
    # =====================================================================
    print("\n💾 STEP 8: Exporting Results")
    print("-" * 70)

    # Save CSV
    output_csv = f"{cfg['paths']['output_dir']}/ndvi_results.csv"
    df = save_csv(results, output_csv)
    print(f"  ✓ CSV saved: {valid_count} records")

    # Generate and save summary statistics
    print("\n  → Generating summary statistics...")
    summary = generate_summary_statistics(df, cfg["h3"]["resolution"])
    summary_path = f"{cfg['paths']['output_dir']}/summary_statistics.json"
    save_summary(summary, summary_path)

    # Create and save metadata
    print("  → Creating metadata...")
    end_time = datetime.now()

    metadata = AnalysisMetadata(
        h3_resolution=cfg["h3"]["resolution"],
        thresholds=thresholds,
        min_pixel_ratio=cfg["h3"]["min_pixel_ratio"],
        processing_date=end_time,
        source_files=source_files,
        total_hexagons_generated=len(hexes),
        valid_results=valid_count,
        invalid_results=invalid_count,
        use_adaptive_severity=use_adaptive,
        n_workers=n_workers,
        avg_confidence=df["confidence"].mean() if "confidence" in df.columns else 0.0,
        avg_pixel_count=(df["pixel_count_2025"].mean() + df["pixel_count_2026"].mean())
        / 2,
        warnings_count=len(warnings),
        morans_i=spatial_stats.get("morans_i"),
        spatial_interpretation=spatial_stats.get("interpretation"),
    )

    metadata_path = f"{cfg['paths']['output_dir']}/analysis_metadata.json"
    save_metadata(metadata, metadata_path)

    # =====================================================================
    # STEP 9: Create Visualizations
    # =====================================================================
    print("\n🎨 STEP 9: Creating Visualizations")
    print("-" * 70)

    print("  → Creating static matplotlib visualization...")
    create_matplotlib_visualization(
        df, f"{cfg['paths']['output_dir']}/ndvi_analysis.png"
    )

    print("  → Creating interactive Folium map...")
    create_folium_map(df, f"{cfg['paths']['output_dir']}/ndvi_change_map.html")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    processing_time = (end_time - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("  ✅ ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\n📊 Summary:")
    print(f"  • Total hexagons: {len(hexes)}")
    print(f"  • Valid results: {valid_count} ({valid_count / len(hexes) * 100:.1f}%)")
    print(f"  • Vegetated hexagons: {summary.vegetated_hexagons}")
    print(f"  • Mean NDVI change: {summary.mean_change:+.4f}")
    print(f"  • Severe changes: {summary.severe_count}")
    print(f"  • Processing time: {processing_time:.1f} seconds")

    print("\n📁 Output files:")
    print(f"  • CSV: {output_csv}")
    print(f"  • Summary: {summary_path}")
    print(f"  • Metadata: {metadata_path}")
    print(f"  • Static plot: {cfg['paths']['output_dir']}/ndvi_analysis.png")
    print(f"  • Interactive map: {cfg['paths']['output_dir']}/ndvi_change_map.html")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
