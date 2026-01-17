# ======= core/h3_sampling.py =======

import warnings
from concurrent.futures import ThreadPoolExecutor

import h3
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import rowcol
from rasterio.windows import Window

# from scipy import stats
from shapely.geometry import Polygon, mapping

from .models import HexResult

# ---------------------------------------------------------------------
# TRANSFORMER CACHE
# ---------------------------------------------------------------------
_TRANSFORMER_CACHE = {}
_BOUNDARY_CACHE = {}


def get_transformer(src, dst):
    key = (src, dst)
    if key not in _TRANSFORMER_CACHE:
        _TRANSFORMER_CACHE[key] = Transformer.from_crs(src, dst, always_xy=True)
    return _TRANSFORMER_CACHE[key]


def get_cached_boundary(hex_id):
    """Cache hexagon boundaries to avoid repeated conversions."""
    if hex_id not in _BOUNDARY_CACHE:
        _BOUNDARY_CACHE[hex_id] = list(h3.cell_to_boundary(hex_id))
    return _BOUNDARY_CACHE[hex_id]


# ---------------------------------------------------------------------
# H3 GRID GENERATION
# ---------------------------------------------------------------------
def generate_h3_grid(bounds, crs, resolution):
    """
    Generate H3 cells covering raster bounds.
    """
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    minx, miny, maxx, maxy = bounds

    # Convert raster bounds to lat/lon
    lonlat = [
        transformer.transform(minx, miny),
        transformer.transform(maxx, miny),
        transformer.transform(maxx, maxy),
        transformer.transform(minx, maxy),
    ]

    # H3 expects (lat, lon)
    latlng = [(lat, lon) for lon, lat in lonlat]
    poly = h3.LatLngPoly(latlng)

    return list(h3.polygon_to_cells(poly, resolution))


# ---------------------------------------------------------------------
# STATISTICS & VALIDATION
# ---------------------------------------------------------------------
def calculate_confidence(pixel_count, std_dev, max_std):
    """
    Calculate confidence score based on sample size and variance.
    Returns: float between 0-1 (1 = high confidence)
    """
    # Confidence increases with more pixels
    sample_confidence = min(pixel_count / 100, 1.0)

    # Confidence decreases with higher variance
    variance_confidence = 1.0 - (std_dev / max_std if max_std > 0 else 0)

    return sample_confidence * 0.6 + variance_confidence * 0.4


def calculate_z_score(value, values_array):
    """Calculate how many standard deviations from mean."""
    if len(values_array) < 2:
        return 0.0
    mean = np.mean(values_array)
    std = np.std(values_array)
    if std == 0:
        return 0.0
    return (value - mean) / std


def classify_area_type(ndvi_2025, ndvi_2026, change):
    """
    Intelligent classification including urban greening/browning.
    """
    if ndvi_2025 < 0.1 and ndvi_2026 < 0.1:
        return "bare_soil_water", False
    elif ndvi_2025 < 0.15 and change > 0.15:
        return "new_vegetation", True  # Urban greening
    elif ndvi_2025 > 0.3 and ndvi_2026 < 0.15:
        return "vegetation_loss", True  # Deforestation/urbanization
    elif ndvi_2025 > 0.15 or ndvi_2026 > 0.15:
        return "vegetated", True
    else:
        return "non_vegetated", False


def adaptive_severity(change, local_changes):
    """
    Classify severity based on local context rather than fixed thresholds.
    """
    if len(local_changes) < 10:
        # Fall back to absolute thresholds
        if abs(change) > 0.3:
            return "Severe"
        elif abs(change) > 0.15:
            return "Moderate"
        else:
            return "Mild"

    # Use percentile-based classification
    abs_changes = np.abs(local_changes)
    percentile_75 = np.percentile(abs_changes, 75)
    percentile_50 = np.percentile(abs_changes, 50)

    abs_change = abs(change)
    if abs_change > percentile_75:
        return "Severe"
    elif abs_change > percentile_50:
        return "Moderate"
    else:
        return "Mild"


# ---------------------------------------------------------------------
# CORE SAMPLING FUNCTION (ENHANCED VERSION)
# ---------------------------------------------------------------------
def sample_hex(
    hex_id,
    ndvi_a,
    ndvi_b,
    meta,
    thresholds,
    min_pixel_ratio,
    global_changes=None,
    max_global_std=None,
):
    """
    Enhanced NDVI sampling with confidence metrics and adaptive classification.

    Args:
        hex_id: H3 hexagon ID
        ndvi_a: 2025 NDVI array
        ndvi_b: 2026 NDVI array
        meta: RasterMeta object
        thresholds: Dict with classification thresholds
        min_pixel_ratio: Minimum pixel coverage ratio
        global_changes: Array of all changes for adaptive severity (optional)
        max_global_std: Maximum std across all hexagons (optional)
    """

    # Convert H3 boundary from lat/lon → raster CRS
    transformer = get_transformer("EPSG:4326", meta.crs)

    # Use cached boundary
    boundary = get_cached_boundary(hex_id)

    coords = []
    for lat, lon in boundary:
        x, y = transformer.transform(lon, lat)
        coords.append((x, y))

    poly = Polygon(coords)
    minx, miny, maxx, maxy = poly.bounds

    # Convert polygon bounds to raster rows/cols
    r0, c0 = rowcol(meta.transform, minx, maxy)
    r1, c1 = rowcol(meta.transform, maxx, miny)

    # Clip to raster extent
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(meta.shape[0], r1 + 1), min(meta.shape[1], c1 + 1)

    if r0 >= r1 or c0 >= c1:
        return None

    # Crop NDVI arrays
    sub_a = ndvi_a[r0:r1, c0:c1]
    sub_b = ndvi_b[r0:r1, c0:c1]

    # Correct transform for the cropped window
    window = Window(c0, r0, c1 - c0, r1 - r0)
    window_transform = rasterio.windows.transform(window, meta.transform)

    # Create mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = geometry_mask(
            [mapping(poly)],
            out_shape=sub_a.shape,
            transform=window_transform,
            invert=True,
        )

    # Apply mask and clean data
    a = sub_a[mask]
    b = sub_b[mask]

    # Remove NaNs and zeros (common in NDVI rasters)
    valid_a = ~np.isnan(a) & (a != 0)
    valid_b = ~np.isnan(b) & (b != 0)

    a = a[valid_a]
    b = b[valid_b]

    if len(a) == 0 or len(b) == 0:
        return None

    # Pixel coverage check
    ratio = min(len(a), len(b)) / max(len(a), len(b))
    if ratio < min_pixel_ratio:
        return None

    # Statistics
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    std_a = float(np.std(a))
    std_b = float(np.std(b))

    change = float(mean_b - mean_a)

    # Calculate confidence
    avg_pixel_count = (len(a) + len(b)) / 2
    avg_std = (std_a + std_b) / 2
    confidence = calculate_confidence(
        avg_pixel_count, avg_std, max_global_std if max_global_std else 0.3
    )

    # Intelligent area classification
    area_type, is_veg = classify_area_type(mean_a, mean_b, change)

    # Adaptive severity classification
    if global_changes is not None and len(global_changes) > 0:
        severity = adaptive_severity(change, global_changes)
    else:
        # Fallback to absolute thresholds
        severity = (
            "Severe"
            if abs(change) > thresholds.get("severe", 0.3)
            else "Moderate"
            if abs(change) > thresholds.get("moderate", 0.15)
            else "Mild"
        )

    # Direction
    direction = (
        "Increase" if change > 0.01 else "Decrease" if change < -0.01 else "Stable"
    )

    # Cell center in lat/lon
    lat, lon = h3.cell_to_latlng(hex_id)

    # Relative percent change
    relative_change = (change / abs(mean_a) * 100) if mean_a != 0 else 0.0

    # Anomaly score (z-score if global changes available)
    anomaly_score = 0.0
    if global_changes is not None and len(global_changes) > 0:
        anomaly_score = float(calculate_z_score(change, global_changes))

    # Uncertainty flag
    uncertainty_flag = (confidence < 0.5) or (len(a) < 20) or (len(b) < 20)

    return HexResult(
        hex_id=hex_id,
        lat=lat,
        lon=lon,
        ndvi_2025=mean_a,
        ndvi_2026=mean_b,
        ndvi_2025_min=float(np.min(a)),
        ndvi_2026_min=float(np.min(b)),
        ndvi_2025_max=float(np.max(a)),
        ndvi_2026_max=float(np.max(b)),
        change=change,
        relative_change=relative_change,
        std_2025=std_a,
        std_2026=std_b,
        pixel_count_2025=len(a),
        pixel_count_2026=len(b),
        is_vegetated=is_veg,
        severity=severity if is_veg else None,  # ✅ None for non-vegetated
        direction=direction if is_veg else None,  # ✅ None for non-vegetated
        area_type=area_type,
        confidence=confidence,
        anomaly_score=anomaly_score,
        uncertainty_flag=uncertainty_flag,
    )


# ---------------------------------------------------------------------
# BATCH PROCESSING WITH THREADING (MEMORY SAFE)
# ---------------------------------------------------------------------
def sample_hexagons_batch(
    hex_ids,
    ndvi_a,
    ndvi_b,
    meta,
    thresholds,
    min_pixel_ratio,
    n_workers=4,
    use_adaptive=True,
):
    """
    Process hexagons using ThreadPoolExecutor (memory-safe alternative to multiprocessing).

    Args:
        hex_ids: List of H3 hexagon IDs
        ndvi_a: 2025 NDVI array
        ndvi_b: 2026 NDVI array
        meta: RasterMeta object
        thresholds: Classification thresholds
        min_pixel_ratio: Minimum pixel coverage
        n_workers: Number of threads (default: 4)
        use_adaptive: Use adaptive severity classification

    Returns:
        List of HexResult objects
    """
    from tqdm import tqdm

    print(f"Processing {len(hex_ids)} hexagons with {n_workers} threads...")

    # First pass: collect initial results for adaptive thresholds
    if use_adaptive:
        print("  → First pass: collecting statistics...")
        initial_results = []

        for hex_id in tqdm(hex_ids, desc="    Sampling", ncols=70):
            result = sample_hex(
                hex_id, ndvi_a, ndvi_b, meta, thresholds, min_pixel_ratio
            )
            if result is not None:
                initial_results.append(result)

        # Extract changes for adaptive classification
        global_changes = np.array([r.change for r in initial_results])
        max_global_std = max(
            [r.std_2025 for r in initial_results]
            + [r.std_2026 for r in initial_results]
        )

        print(
            f"  → Second pass: adaptive classification on {len(initial_results)} valid hexagons..."
        )

        # Second pass with adaptive thresholds using threading
        def process_with_adaptive(hex_id):
            return sample_hex(
                hex_id,
                ndvi_a,
                ndvi_b,
                meta,
                thresholds,
                min_pixel_ratio,
                global_changes,
                max_global_std,
            )

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_with_adaptive, hex_ids),
                    total=len(hex_ids),
                    desc="    Classifying",
                    ncols=70,
                )
            )

    else:
        # Single pass with threading
        def process_basic(hex_id):
            return sample_hex(hex_id, ndvi_a, ndvi_b, meta, thresholds, min_pixel_ratio)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_basic, hex_ids),
                    total=len(hex_ids),
                    desc="    Sampling",
                    ncols=70,
                )
            )

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    print(f"  ✓ Completed: {len(valid_results)}/{len(hex_ids)} hexagons valid")

    return valid_results


# ---------------------------------------------------------------------
# VALIDATION & QUALITY CHECKS
# ---------------------------------------------------------------------
def validate_results(results):
    """
    Validate results and return warnings for suspicious data.
    """
    warnings_list = []

    for r in results:
        # Unrealistic NDVI values
        if abs(r.ndvi_2025) > 1.0 or abs(r.ndvi_2026) > 1.0:
            warnings_list.append(
                f"Invalid NDVI at {r.hex_id[:12]}: "
                + f"2025={r.ndvi_2025:.3f}, 2026={r.ndvi_2026:.3f}"
            )

        # Too few pixels (unreliable)
        if r.pixel_count_2025 < 10 or r.pixel_count_2026 < 10:
            warnings_list.append(
                f"Low sample size at {r.hex_id[:12]}: "
                + f"{r.pixel_count_2025}/{r.pixel_count_2026} pixels"
            )

        # Extreme changes (possible data error)
        if abs(r.change) > 0.8:
            warnings_list.append(f"Extreme change at {r.hex_id[:12]}: {r.change:+.3f}")

        # High uncertainty
        if r.uncertainty_flag:
            warnings_list.append(
                f"High uncertainty at {r.hex_id[:12]}: confidence={r.confidence:.2f}"
            )

    return warnings_list


# ---------------------------------------------------------------------
# SPATIAL AUTOCORRELATION (MORAN'S I)
# ---------------------------------------------------------------------
def calculate_spatial_autocorrelation(results):
    """
    Calculate Moran's I to detect spatial clustering.
    Returns: dict with Moran's I statistic and p-value
    """
    if len(results) < 10:
        return {
            "morans_i": None,
            "p_value": None,
            "interpretation": "Insufficient data",
        }

    # Build spatial weights matrix (simple: hexagons are neighbors if they share edge)
    hex_ids = [r.hex_id for r in results]
    changes = np.array([r.change for r in results])

    n = len(hex_ids)
    W = np.zeros((n, n))

    for i, hex_i in enumerate(hex_ids):
        neighbors = h3.grid_disk(hex_i, 1)
        for j, hex_j in enumerate(hex_ids):
            if hex_j in neighbors and i != j:
                W[i, j] = 1

    # Normalize weights
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums[:, np.newaxis]

    # Calculate Moran's I
    mean_change = changes.mean()
    deviations = changes - mean_change

    numerator = np.sum(W * np.outer(deviations, deviations))
    denominator = np.sum(deviations**2)

    morans_i = (n / W.sum()) * (numerator / denominator) if denominator > 0 else 0

    # Interpretation
    if morans_i > 0.3:
        interpretation = "Strong positive clustering (hotspots)"
    elif morans_i > 0.1:
        interpretation = "Moderate positive clustering"
    elif morans_i < -0.1:
        interpretation = "Negative clustering (dispersed)"
    else:
        interpretation = "Random distribution"

    return {
        "morans_i": float(morans_i),
        "interpretation": interpretation,
        "n_hexagons": n,
    }
