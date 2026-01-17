from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from affine import Affine
from pyproj.crs import CRS
from rasterio.coords import BoundingBox


@dataclass
class RasterMeta:
    transform: Affine
    crs: CRS
    bounds: BoundingBox
    shape: tuple


@dataclass
class NDVIStats:
    mean: float
    std: float
    min: float
    max: float
    count: int


@dataclass
class HexResult:
    """Enhanced hexagon result with confidence and analysis metrics."""

    hex_id: str
    lat: float
    lon: float
    ndvi_2025: float
    ndvi_2026: float
    ndvi_2025_min: float
    ndvi_2026_min: float
    ndvi_2025_max: float
    ndvi_2026_max: float
    change: float
    relative_change: float
    std_2025: float
    std_2026: float
    pixel_count_2025: int
    pixel_count_2026: int
    is_vegetated: bool
    severity: Optional[str]  # None for non-vegetated areas
    direction: Optional[str]  # None for non-vegetated areas

    area_type: (
        str  # "vegetated", "bare_soil_water", "new_vegetation", "vegetation_loss"
    )
    confidence: float  # 0-1 confidence score
    anomaly_score: float  # Z-score indicating how unusual the change is
    uncertainty_flag: bool  # True if low confidence/sample size


@dataclass
class AnalysisMetadata:
    """Metadata for reproducibility and quality tracking."""

    h3_resolution: int
    thresholds: dict
    min_pixel_ratio: float
    processing_date: datetime
    source_files: dict
    total_hexagons_generated: int
    valid_results: int
    invalid_results: int
    use_adaptive_severity: bool
    n_workers: int

    # Quality metrics
    avg_confidence: float
    avg_pixel_count: float
    warnings_count: int

    # Spatial statistics
    morans_i: Optional[float] = None
    spatial_interpretation: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary for JSON export."""
        return {
            "h3_resolution": self.h3_resolution,
            "thresholds": self.thresholds,
            "min_pixel_ratio": self.min_pixel_ratio,
            "processing_date": self.processing_date.isoformat(),
            "source_files": self.source_files,
            "total_hexagons_generated": self.total_hexagons_generated,
            "valid_results": self.valid_results,
            "invalid_results": self.invalid_results,
            "success_rate": f"{self.valid_results / self.total_hexagons_generated * 100:.1f}%",
            "use_adaptive_severity": self.use_adaptive_severity,
            "n_workers": self.n_workers,
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_pixel_count": round(self.avg_pixel_count, 1),
            "warnings_count": self.warnings_count,
            "morans_i": round(self.morans_i, 3) if self.morans_i else None,
            "spatial_interpretation": self.spatial_interpretation,
        }


@dataclass
class SummaryStatistics:
    """Aggregate statistics for the entire analysis."""

    total_hexagons: int
    vegetated_hexagons: int
    non_vegetated_hexagons: int

    # Change statistics
    mean_change: float
    median_change: float
    std_change: float
    min_change: float
    max_change: float

    # Direction breakdown
    increase_count: int
    decrease_count: int
    stable_count: int

    # Severity breakdown
    severe_count: int
    moderate_count: int
    mild_count: int

    # Area breakdown
    total_area_km2: float
    vegetated_area_km2: float
    severe_change_area_km2: float

    # Quality metrics
    avg_confidence: float
    high_uncertainty_count: int

    def to_dict(self):
        """Convert to dictionary for export."""
        return {
            "total_hexagons": self.total_hexagons,
            "vegetated_hexagons": self.vegetated_hexagons,
            "non_vegetated_hexagons": self.non_vegetated_hexagons,
            "vegetation_percentage": round(
                self.vegetated_hexagons / self.total_hexagons * 100, 1
            ),
            "change_statistics": {
                "mean": round(self.mean_change, 4),
                "median": round(self.median_change, 4),
                "std": round(self.std_change, 4),
                "min": round(self.min_change, 4),
                "max": round(self.max_change, 4),
            },
            "direction_breakdown": {
                "increase": self.increase_count,
                "decrease": self.decrease_count,
                "stable": self.stable_count,
            },
            "severity_breakdown": {
                "severe": self.severe_count,
                "moderate": self.moderate_count,
                "mild": self.mild_count,
            },
            "area_statistics_km2": {
                "total": round(self.total_area_km2, 2),
                "vegetated": round(self.vegetated_area_km2, 2),
                "severe_change": round(self.severe_change_area_km2, 2),
            },
            "quality_metrics": {
                "avg_confidence": round(self.avg_confidence, 3),
                "high_uncertainty_count": self.high_uncertainty_count,
                "high_uncertainty_percentage": round(
                    self.high_uncertainty_count / self.total_hexagons * 100, 1
                ),
            },
        }
