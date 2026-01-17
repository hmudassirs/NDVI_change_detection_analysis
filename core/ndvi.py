import warnings

import numpy as np


def compute_ndvi(red: np.ndarray, nir: np.ndarray, nodata_value=None) -> np.ndarray:
    """
    Compute NDVI safely with comprehensive data validation.

    NDVI = (NIR - RED) / (NIR + RED)

    Args:
        red: Red band array
        nir: Near-infrared band array
        nodata_value: Optional nodata value to mask (default: None)

    Returns:
        NDVI array with values in range [-1, 1]
    """

    # Convert to float32
    red = red.astype("float32")
    nir = nir.astype("float32")

    # Create output array filled with NaN
    ndvi = np.full_like(red, np.nan, dtype="float32")

    # Build valid data mask
    valid_mask = np.ones(red.shape, dtype=bool)

    # Mask NaN values
    valid_mask &= ~np.isnan(red)
    valid_mask &= ~np.isnan(nir)

    # Mask infinite values
    valid_mask &= ~np.isinf(red)
    valid_mask &= ~np.isinf(nir)

    # Mask nodata values if specified
    if nodata_value is not None:
        valid_mask &= red != nodata_value
        valid_mask &= nir != nodata_value

    # Mask very small values (likely noise or nodata)
    # Sentinel-2 reflectance values should be > 0
    valid_mask &= red > 0
    valid_mask &= nir > 0

    # Mask unrealistic reflectance values (Sentinel-2 L2A should be 0-10000)
    # Allow up to 15000 for safety margin
    valid_mask &= red < 15000
    valid_mask &= nir < 15000

    # Calculate denominator
    denom = nir + red

    # Mask zero denominator (though this should be rare with above filters)
    valid_mask &= denom != 0
    valid_mask &= ~np.isnan(denom)

    # Count valid pixels
    n_valid = np.count_nonzero(valid_mask)
    n_total = red.size

    if n_valid == 0:
        warnings.warn(
            "No valid pixels found for NDVI computation. "
            + "All pixels are NaN, zero, negative, or exceed threshold.",
            RuntimeWarning,
        )
        return ndvi

    # Compute NDVI for valid pixels only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ndvi[valid_mask] = (nir[valid_mask] - red[valid_mask]) / denom[valid_mask]

    # Clip to valid NDVI range
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Print statistics
    valid_percent = (n_valid / n_total) * 100
    print(f"    Valid pixels: {n_valid:,} / {n_total:,} ({valid_percent:.1f}%)")

    # Calculate statistics on valid data only
    valid_ndvi = ndvi[valid_mask]
    if len(valid_ndvi) > 0:
        ndvi_min = np.nanmin(valid_ndvi)
        ndvi_max = np.nanmax(valid_ndvi)
        ndvi_mean = np.nanmean(valid_ndvi)
        print(
            f"    NDVI range: [{ndvi_min:.3f}, {ndvi_max:.3f}], mean: {ndvi_mean:.3f}"
        )

    return ndvi


def validate_band(band: np.ndarray, band_name: str) -> dict:
    """
    Validate a raster band and return statistics.

    Args:
        band: Input raster array
        band_name: Name of the band (for reporting)

    Returns:
        dict: Validation statistics
    """
    stats = {
        "name": band_name,
        "shape": band.shape,
        "dtype": band.dtype,
        "total_pixels": band.size,
        "nan_count": np.count_nonzero(np.isnan(band)),
        "inf_count": np.count_nonzero(np.isinf(band)),
        "zero_count": np.count_nonzero(band == 0),
        "negative_count": np.count_nonzero(band < 0),
        "min": np.nanmin(band) if not np.all(np.isnan(band)) else np.nan,
        "max": np.nanmax(band) if not np.all(np.isnan(band)) else np.nan,
        "mean": np.nanmean(band) if not np.all(np.isnan(band)) else np.nan,
    }

    # Calculate valid pixel percentage
    valid_pixels = (
        stats["total_pixels"]
        - stats["nan_count"]
        - stats["inf_count"]
        - stats["zero_count"]
        - stats["negative_count"]
    )
    stats["valid_pixels"] = valid_pixels
    stats["valid_percent"] = (valid_pixels / stats["total_pixels"]) * 100

    return stats


def print_band_validation(stats: dict):
    """Print band validation statistics."""
    print(f"\n  {stats['name']}:")
    print(f"    Shape: {stats['shape']}, Type: {stats['dtype']}")
    print(f"    Total pixels: {stats['total_pixels']:,}")
    print(
        f"    Valid pixels: {stats['valid_pixels']:,} ({stats['valid_percent']:.1f}%)"
    )

    if stats["nan_count"] > 0:
        print(f"    ⚠️  NaN values: {stats['nan_count']:,}")
    if stats["inf_count"] > 0:
        print(f"    ⚠️  Infinite values: {stats['inf_count']:,}")
    if stats["zero_count"] > 0:
        print(f"    ⚠️  Zero values: {stats['zero_count']:,}")
    if stats["negative_count"] > 0:
        print(f"    ⚠️  Negative values: {stats['negative_count']:,}")

    if not np.isnan(stats["mean"]):
        print(
            f"    Value range: [{stats['min']:.1f}, {stats['max']:.1f}], mean: {stats['mean']:.1f}"
        )
    else:
        print("    ⚠️  All values are NaN!")


def diagnose_bands(red: np.ndarray, nir: np.ndarray):
    """
    Diagnose issues with input bands.

    Args:
        red: Red band array
        nir: NIR band array
    """
    print("\n🔍 Band Diagnostics:")
    print("-" * 70)

    red_stats = validate_band(red, "Red Band (B04)")
    nir_stats = validate_band(nir, "NIR Band (B08)")

    print_band_validation(red_stats)
    print_band_validation(nir_stats)

    # Check if bands are in expected range
    print("\n  Range Check:")

    # Sentinel-2 L2A Surface Reflectance range: 0-10000
    expected_min = 0
    expected_max = 10000

    for stats in [red_stats, nir_stats]:
        if not np.isnan(stats["min"]) and not np.isnan(stats["max"]):
            if stats["min"] < expected_min or stats["max"] > expected_max:
                print(f"    ⚠️  {stats['name']} outside expected range [0, 10000]")
                print(f"       Actual range: [{stats['min']:.1f}, {stats['max']:.1f}]")

                # Provide suggestions
                if stats["max"] > 10000 and stats["max"] < 1:
                    print(
                        "       💡 Values appear to be in 0-1 range (already normalized)"
                    )
                elif stats["max"] > 10000:
                    print(
                        "       💡 Values may need scaling or have incorrect nodata handling"
                    )
            else:
                print(f"    ✓ {stats['name']} in expected range")

    # Check for common issues
    print("\n  Common Issues:")
    issues_found = False

    if red_stats["valid_percent"] < 50 or nir_stats["valid_percent"] < 50:
        print(
            "    ⚠️  Less than 50% valid pixels - check nodata handling in raster loading"
        )
        issues_found = True

    if red_stats["nan_count"] > 0 or nir_stats["nan_count"] > 0:
        print("    ⚠️  NaN values present - ensure proper nodata value handling")
        issues_found = True

    if red_stats["zero_count"] > red_stats["total_pixels"] * 0.5:
        print("    ⚠️  High percentage of zeros - may indicate nodata coded as 0")
        issues_found = True

    if not issues_found:
        print("    ✓ No obvious issues detected")

    print("-" * 70)
