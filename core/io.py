import os

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .models import RasterMeta


def find_band(path, pattern):
    """
    Find a band file by pattern in directory.

    Args:
        path: Directory to search
        pattern: Band identifier (e.g., 'B04', 'B08')

    Returns:
        str: Full path to band file

    Raises:
        FileNotFoundError: If band not found
    """
    # Support both directory and direct file paths
    if os.path.isfile(path):
        return path

    for root, _, files in os.walk(path):
        for f in files:
            # Match pattern in filename
            if pattern in f and (f.endswith(".tif") or f.endswith(".jp2")):
                return os.path.join(root, f)

    raise FileNotFoundError(f"Band '{pattern}' not found in {path}")


def load_raster(path, scale_factor=None):
    """
    Load raster with proper nodata handling.

    Args:
        path: Path to raster file
        scale_factor: Optional scaling factor (e.g., 10000 for Sentinel-2 L2A)

    Returns:
        tuple: (data array, RasterMeta object)
    """
    print(f"    Loading: {os.path.basename(path)}")

    with rasterio.open(path) as src:
        # Read first band
        data = src.read(1).astype("float32")

        # Get nodata value
        nodata = src.nodata
        print(f"      Nodata value: {nodata}")
        print(f"      Shape: {data.shape}, CRS: {src.crs}")

        # Count initial statistics
        total_pixels = data.size

        # Handle nodata values
        if nodata is not None:
            # Mask nodata values
            nodata_mask = data == nodata
            nodata_count = np.count_nonzero(nodata_mask)
            data[nodata_mask] = np.nan
            print(
                f"      Nodata pixels: {nodata_count:,} ({nodata_count / total_pixels * 100:.1f}%)"
            )

        # Additional cleanup for Sentinel-2 data
        # Sometimes nodata is coded as 0 or very large values

        # Mask zeros if they seem to be nodata (more than 10% of image)
        zero_mask = data == 0
        zero_count = np.count_nonzero(zero_mask)
        if zero_count > total_pixels * 0.1:
            print(
                f"      ⚠️  High zero count: {zero_count:,} ({zero_count / total_pixels * 100:.1f}%)"
            )
            print("         Treating zeros as nodata")
            data[zero_mask] = np.nan

        # Mask negative values (invalid for reflectance)
        negative_mask = data < 0
        negative_count = np.count_nonzero(negative_mask)
        if negative_count > 0:
            print(f"      ⚠️  Negative values: {negative_count:,} - setting to NaN")
            data[negative_mask] = np.nan

        # Detect and handle scale
        data_max = np.nanmax(data)
        # data_min = np.nanmin(data)

        # Sentinel-2 L2A data is typically 0-10000 (scaled reflectance)
        if data_max > 10000:
            print(f"      ⚠️  Values exceed 10000 (max: {data_max:.1f})")
            if data_max < 65535:  # Likely 16-bit scaled incorrectly
                print("         Data may be incorrectly scaled")

        # Apply scaling if provided
        if scale_factor is not None:
            print(f"      Applying scale factor: 1/{scale_factor}")
            data = data / scale_factor

        # Final statistics
        valid_count = np.count_nonzero(~np.isnan(data))
        print(
            f"      Valid pixels: {valid_count:,} ({valid_count / total_pixels * 100:.1f}%)"
        )

        if valid_count > 0:
            print(f"      Value range: [{np.nanmin(data):.1f}, {np.nanmax(data):.1f}]")
        else:
            print("      ⚠️  WARNING: No valid pixels found!")

        # Create metadata object
        meta = RasterMeta(
            transform=src.transform, crs=src.crs, bounds=src.bounds, shape=data.shape
        )

    return data, meta


def align_raster(reference_meta, target_data, target_meta):
    """
    Align target raster to reference grid.

    Args:
        reference_meta: RasterMeta of reference
        target_data: Target data array
        target_meta: RasterMeta of target

    Returns:
        np.ndarray: Aligned raster
    """
    # Check if already aligned
    if (
        reference_meta.shape == target_meta.shape
        and reference_meta.transform == target_meta.transform
        and reference_meta.crs == target_meta.crs
    ):
        print("      Rasters already aligned")
        return target_data

    print(f"      Aligning: {target_meta.shape} -> {reference_meta.shape}")

    # Create output array
    aligned = np.empty(reference_meta.shape, dtype="float32")
    aligned.fill(np.nan)  # Initialize with NaN

    # Reproject
    reproject(
        target_data,
        aligned,
        src_transform=target_meta.transform,
        src_crs=target_meta.crs,
        dst_transform=reference_meta.transform,
        dst_crs=reference_meta.crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    # Check alignment result
    valid_aligned = np.count_nonzero(~np.isnan(aligned))
    print(
        f"      Aligned valid pixels: {valid_aligned:,} ({valid_aligned / aligned.size * 100:.1f}%)"
    )

    return aligned
