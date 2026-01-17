"""
Sentinel-2 Dataset Preparation Script
Extracts and processes 10m resolution bands (B02, B03, B04, B08) for study area.
"""

import glob
import json
import os
import tempfile
from pathlib import Path

import pyproj
import rasterio

# from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.merge import merge
from shapely.geometry import box, shape
from shapely.ops import transform
from tqdm import tqdm

# =====================================================================
# CONFIGURATION
# =====================================================================

# Required 10m resolution bands for NDVI calculation
REQUIRED_BANDS_10M = ["B02", "B03", "B04", "B08"]

# Band descriptions
BAND_INFO = {
    "B02": "Blue (490 nm)",
    "B03": "Green (560 nm)",
    "B04": "Red (665 nm)",
    "B08": "NIR (842 nm)",
}


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================


def load_study_area(geojson_path):
    """Load the study area polygon from GeoJSON file."""
    print(f"  → Loading study area from: {geojson_path}")

    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    # Get the first feature's geometry
    geometry = geojson["features"][0]["geometry"]
    print(f"    ✓ Study area loaded (type: {geometry['type']})")

    return geometry


def reproject_geometry(geom, src_crs, dst_crs):
    """Reproject a geometry from source CRS to destination CRS."""
    if src_crs == dst_crs:
        return geom

    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    geom_shape = shape(geom)
    reprojected = transform(project, geom_shape)
    return reprojected.__geo_interface__


def get_band_files(safe_dir, required_bands=REQUIRED_BANDS_10M):
    """
    Get 10m resolution band files from a SAFE directory.

    Args:
        safe_dir: Path to SAFE directory
        required_bands: List of band names to extract (default: 10m bands)

    Returns:
        dict: {band_name: file_path}
    """
    print(f"\n    Searching in: {os.path.basename(safe_dir)}")

    # Check if directory exists
    if not os.path.exists(safe_dir):
        print("      ⚠️  Directory does not exist!")
        return {}

    # Try multiple possible paths for Sentinel-2 data structure
    possible_paths = [
        Path(safe_dir) / "GRANULE" / "*" / "IMG_DATA" / "R10m" / "*.jp2",
        Path(safe_dir) / "GRANULE" / "*" / "IMG_DATA" / "*.jp2",
        Path(safe_dir) / "IMG_DATA" / "R10m" / "*.jp2",
        Path(safe_dir) / "*.jp2",
        Path(safe_dir) / "*.tif",  # Support GeoTIFF format
    ]

    band_files = {}
    all_files = []

    # Collect all matching files
    for path_pattern in possible_paths:
        files = glob.glob(str(path_pattern))
        if files:
            all_files.extend(files)

    if not all_files:
        print("      ⚠️  No image files found")
        return {}

    # Extract required bands
    for band_name in required_bands:
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # Match band identifier in filename (e.g., _B04_, _B08_)
            if f"_{band_name}_" in filename or f"_{band_name}." in filename:
                band_files[band_name] = file_path
                print(f"      ✓ Found {band_name}: {filename}")
                break

    # Check if all required bands were found
    missing_bands = set(required_bands) - set(band_files.keys())
    if missing_bands:
        print(f"      ⚠️  Missing bands: {', '.join(missing_bands)}")

    return band_files


def check_overlap(raster_path, geom, src_crs="EPSG:4326"):
    """Check if geometry overlaps with raster bounds."""
    with rasterio.open(raster_path) as src:
        # Reproject geometry to raster CRS
        geom_reproj = reproject_geometry(geom, src_crs, src.crs)
        geom_shape = shape(geom_reproj)

        # Get raster bounds
        raster_bounds = box(*src.bounds)

        # Check intersection
        return geom_shape.intersects(raster_bounds)


# =====================================================================
# MAIN PROCESSING FUNCTIONS
# =====================================================================


def process_single_band(band_files_list, band_name, study_area_geom, output_path):
    """
    Process a single band: merge if multiple tiles, clip to study area.

    Args:
        band_files_list: List of file paths for this band (from different tiles)
        band_name: Band identifier (e.g., 'B04')
        study_area_geom: Study area geometry for clipping
        output_path: Output file path

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Filter files that overlap with study area
        overlapping_files = []
        for band_file in band_files_list:
            if check_overlap(band_file, study_area_geom):
                overlapping_files.append(band_file)

        if not overlapping_files:
            print(f"      ⚠️  No overlapping tiles for {band_name}")
            return False

        # Use only overlapping files
        band_files_list = overlapping_files

        if len(band_files_list) == 1:
            # Single file - just extract study area
            print("      → Processing single tile...")
            with rasterio.open(band_files_list[0]) as src:
                # Reproject geometry to match raster CRS
                geom_reproj = reproject_geometry(study_area_geom, "EPSG:4326", src.crs)

                out_image, out_transform = mask(src, [geom_reproj], crop=True)
                out_meta = src.meta.copy()

        else:
            # Multiple files - merge with 'last' method (uses latest image for overlap)
            print(f"      → Merging {len(band_files_list)} tiles...")
            src_files = [rasterio.open(f) for f in band_files_list]

            # Merge the rasters
            mosaic, out_transform = merge(src_files, method="last")

            # Get metadata from first source
            out_meta = src_files[0].meta.copy()
            target_crs = src_files[0].crs

            # Close source files
            for src in src_files:
                src.close()

            # Create temporary mosaic file
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"temp_mosaic_{band_name}.tif")

            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                }
            )

            with rasterio.open(temp_path, "w", **out_meta) as dest:
                dest.write(mosaic)

            # Clip the mosaic to study area
            with rasterio.open(temp_path) as src:
                geom_reproj = reproject_geometry(
                    study_area_geom, "EPSG:4326", target_crs
                )
                out_image, out_transform = mask(src, [geom_reproj], crop=True)
                out_meta = src.meta.copy()

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Update metadata for the clipped output
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
            }
        )

        # Write the clipped band
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"      ✓ Saved: {band_name}.tif")
        print(f"        Shape: {out_image.shape}, CRS: {out_meta['crs']}")

        return True

    except Exception as e:
        print(f"      ✗ Error processing {band_name}: {str(e)}")
        return False


def process_dataset(safe_dirs, output_dir, study_area_geom, dataset_name):
    """
    Process multiple SAFE datasets and extract 10m bands for study area.

    Args:
        safe_dirs: List of SAFE directory paths
        output_dir: Output directory for processed bands
        study_area_geom: Study area geometry for clipping
        dataset_name: Name for this dataset (e.g., "Jan 2025")
    """
    print(f"\n{'=' * 70}")
    print(f"  Processing {dataset_name}")
    print(f"{'=' * 70}")

    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to collect all band files across datasets
    # Structure: {band_name: [file1, file2, ...]}
    all_band_files = {band: [] for band in REQUIRED_BANDS_10M}

    # Collect band files from all SAFE directories
    print(f"\n  📁 Scanning {len(safe_dirs)} SAFE dataset(s)...")

    for safe_dir in safe_dirs:
        band_files = get_band_files(safe_dir, REQUIRED_BANDS_10M)

        for band_name, file_path in band_files.items():
            all_band_files[band_name].append(file_path)

    # Summary of found bands
    print("\n  📊 Band Summary:")
    for band_name in REQUIRED_BANDS_10M:
        count = len(all_band_files[band_name])
        status = "✓" if count > 0 else "✗"
        print(f"    {status} {band_name} ({BAND_INFO[band_name]}): {count} file(s)")

    # Process each band
    print("\n  🔧 Processing bands...")

    success_count = 0

    for band_name in tqdm(REQUIRED_BANDS_10M, desc="  Processing", ncols=70):
        if not all_band_files[band_name]:
            print(f"    ✗ {band_name}: No files found")
            continue

        output_path = os.path.join(output_dir, f"{band_name}.tif")

        success = process_single_band(
            all_band_files[band_name], band_name, study_area_geom, output_path
        )

        if success:
            success_count += 1

    # Final summary
    print(
        f"\n  ✓ Dataset processing complete: {success_count}/{len(REQUIRED_BANDS_10M)} bands saved"
    )
    print(f"    Output directory: {output_dir}")


# =====================================================================
# MAIN FUNCTION
# =====================================================================


def main():
    """Main entry point for dataset preparation."""

    print("\n" + "=" * 70)
    print("  SENTINEL-2 DATASET PREPARATION")
    print("  10m Resolution Bands (B02, B03, B04, B08)")
    print("=" * 70 + "\n")

    # =====================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # =====================================================================

    # Study area GeoJSON
    geojson_path = "Islamabad.geojson"

    # Base directory containing SAFE datasets
    base_dir = r"D:\py\geemaps\ds"

    # Dataset 1: January 2025 (Pre-event)
    jan_2025_dirs = [
        os.path.join(
            base_dir,
            "sentinal_2_ds",
            "Islamabad",
            "S2B_MSIL2A_20250109T055129_N0511_R048_T43SCT_20250109T064515.SAFE",
        ),
    ]

    # Dataset 2: January 2026 (Post-event)
    jan_2026_dirs = [
        os.path.join(
            base_dir,
            "sentinal_2_ds",
            "Islamabad",
            "S2B_MSIL2A_20260114T055059_N0511_R048_T43SCT_20260114T074840.SAFE",
        ),
    ]

    # Output directories in 'dataset' folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "dataset")

    jan_2025_output = os.path.join(dataset_dir, "jan_2025_10m")
    jan_2026_output = os.path.join(dataset_dir, "jan_2026_10m")

    # =====================================================================
    # LOAD STUDY AREA
    # =====================================================================

    print("📍 Loading Study Area")
    print("-" * 70)

    if not os.path.exists(geojson_path):
        print(f"  ✗ Error: GeoJSON file not found: {geojson_path}")
        print("    Please provide a valid study area GeoJSON file.")
        return

    study_area = load_study_area(geojson_path)

    # =====================================================================
    # PROCESS DATASETS
    # =====================================================================

    # Process January 2025
    process_dataset(
        safe_dirs=jan_2025_dirs,
        output_dir=jan_2025_output,
        study_area_geom=study_area,
        dataset_name="January 2025 (Pre-event)",
    )

    # Process January 2026
    process_dataset(
        safe_dirs=jan_2026_dirs,
        output_dir=jan_2026_output,
        study_area_geom=study_area,
        dataset_name="January 2026 (Post-event)",
    )

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================

    print("\n" + "=" * 70)
    print("  ✅ DATASET PREPARATION COMPLETE!")
    print("=" * 70)

    print("\n📁 Output Directories:")
    print(f"  • January 2025: {jan_2025_output}")
    print(f"  • January 2026: {jan_2026_output}")

    print("\n📊 Required Bands (10m resolution):")
    for band_name, description in BAND_INFO.items():
        print(f"  • {band_name}: {description}")

    print("\n💡 Next Steps:")
    print("  1. Update config.yaml with these paths:")
    print(f'     jan_2025: "{jan_2025_output}"')
    print(f'     jan_2026: "{jan_2026_output}"')
    print("  2. Run main.py to perform NDVI analysis")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
