# NDVI Change Detection Analysis

Automated NDVI change detection using Sentinel-2 imagery and H3 hexagonal grids with advanced analytics.

## Features

### 🚀 Performance Enhancements
- **Thread-based parallelization** (memory-safe, no multiprocessing issues)
- **Cached transformations** for coordinate conversions
- **Progress bars** for all long-running operations
- **Batch processing** with optimal resource usage

### 📊 Advanced Analysis
- **Adaptive severity classification** (context-aware thresholds)
- **Confidence scoring** (based on sample size & variance)
- **Spatial autocorrelation** (Moran's I for clustering detection)
- **Anomaly detection** (z-score for unusual changes)
- **Area type classification** (urban greening, vegetation loss, etc.)
- **Quality validation** with automated warnings

### 📍 Smart Non-Vegetated Handling
- Severity & Direction set to `None` for non-vegetated areas
- Tracks important transitions (new vegetation, vegetation loss)
- Distinguishes bare soil, water, and urban areas

## Project Structure

```
project/
├── dataset/                      # ← NEW: Prepared datasets
│   ├── jan_2025_10m/            # January 2025 bands (B02, B03, B04, B08)
│   └── jan_2026_10m/            # January 2026 bands
│
├── core/
│   ├── h3_sampling.py           # Enhanced sampling with threading & analysis
│   ├── models.py                # Enhanced data models with new fields
│   ├── ndvi.py                  # NDVI computation
│   ├── io.py                    # Raster I/O operations
│   └── viz.py                   # Enhanced visualizations
│
├── dataset/
│   └── prepare_dataset.py       # ← NEW: Dataset preparation script
│
├── outputs/                     # Analysis results
│   ├── ndvi_results.csv
│   ├── summary_statistics.json
│   ├── analysis_metadata.json
│   ├── ndvi_analysis.png
│   └── ndvi_change_map.html
│
├── main.py                      # ← Enhanced with progress bars
├── config.yaml                  # Updated configuration
├── requirements.txt             # Updated dependencies
├── Islamabad.geojson           # Study area boundary
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare Dataset

First, download SAFE tiles from https://dataspace.copernicus.eu/
and prepare your Sentinel-2 data by extracting 10m bands for your study area:

```bash
python dataset/prepare_dataset.py
```

**What it does:**
1. Loads study area from GeoJSON
2. Scans SAFE directories for required bands (B02, B03, B04, B08)
3. Merges multiple tiles if needed
4. Clips to study area boundary
5. Saves processed bands to `dataset/` folder

**Configuration** (edit `dataset/prepare_dataset.py`):
```python
# Study area
geojson_path = "Islamabad.geojson"

# Input SAFE directories
jan_2025_dirs = [
    "path/to/S2B_MSIL2A_20250109T055129_..._T43SCT_....SAFE",
]

jan_2026_dirs = [
    "path/to/S2B_MSIL2A_20260114T055059_..._T43SCT_....SAFE",
]
```

### Step 2: Update Configuration

Edit `config.yaml` to match your prepared dataset:

```yaml
paths:
  jan_2025: "dataset/jan_2025_10m"
  jan_2026: "dataset/jan_2026_10m"
  output_dir: "outputs"

h3:
  resolution: 9  # 9 ≈ 0.105 km², 8 ≈ 0.737 km², 7 ≈ 5.16 km²
  min_pixel_ratio: 0.7

processing:
  n_workers: 4  # Adjust based on CPU cores
  use_adaptive_severity: true
```

### Step 3: Run Analysis

```bash
python main.py
```

**Output includes:**
- **CSV**: All hexagon results with statistics
- **Summary JSON**: Aggregate statistics
- **Metadata JSON**: Processing parameters & quality metrics
- **Static plot**: Multi-panel matplotlib visualization
- **Interactive map**: Folium map with detailed popups

## Output Files

### 1. `ndvi_results.csv`
Complete results for each hexagon with all metrics:
- NDVI values (2025 & 2026)
- Change metrics (absolute, relative, z-score)
- Classification (severity, direction, area type)
- Quality metrics (confidence, uncertainty flag)
- Spatial coordinates

### 2. `summary_statistics.json`
Aggregate statistics:
```json
{
  "total_hexagons": 1234,
  "vegetated_hexagons": 856,
  "change_statistics": {
    "mean": -0.0234,
    "median": -0.0189
  },
  "severity_breakdown": {
    "severe": 45,
    "moderate": 123,
    "mild": 688
  },
  "quality_metrics": {
    "avg_confidence": 0.847,
    "high_uncertainty_count": 67
  }
}
```

### 3. `analysis_metadata.json`
Processing metadata for reproducibility:
- Source files
- Thresholds used
- Processing parameters
- Spatial statistics (Moran's I)
- Quality warnings count

### 4. `ndvi_analysis.png`
6-panel static visualization:
1. NDVI 2025
2. NDVI 2026
3. Confidence map
4. NDVI change
5. Change distribution
6. Severity breakdown

### 5. `ndvi_change_map.html`
Interactive map with:
- Multiple base layers (Satellite, OSM, Light)
- Color-coded hexagons by change
- Detailed popups with statistics
- Legend with severity levels

## New Features Explained

### Confidence Scoring
Each hexagon gets a confidence score (0-1) based on:
- **Sample size**: More pixels = higher confidence
- **Variance**: Lower std deviation = higher confidence

Low confidence hexagons are flagged for review.

### Adaptive Severity Classification
Instead of fixed thresholds, severity is determined by:
- **Local context**: Compared to neighboring hexagons
- **Percentile-based**: Uses 50th & 75th percentiles
- **Dynamic thresholds**: Adapts to local change patterns

### Area Type Classification
Intelligently classifies:
- **Vegetated**: Standard vegetation areas
- **New vegetation**: Urban greening (low → high NDVI)
- **Vegetation loss**: Deforestation/urbanization (high → low NDVI)
- **Bare soil/water**: Consistently low NDVI

### Spatial Autocorrelation
Moran's I statistic detects:
- **Clustering**: Changes concentrated in areas
- **Dispersion**: Changes scattered randomly
- **Hotspots**: Identifies areas with similar changes

## Performance Tips

### H3 Resolution Selection
| Resolution | Area per hex | Use case |
|------------|--------------|----------|
| 7 | ~5.16 km² | Regional overview |
| 8 | ~0.74 km² | District level |
| 9 | ~0.11 km² | **Recommended** |
| 10 | ~0.015 km² | Very detailed |

Higher resolution = more hexagons = longer processing time

### Thread Count
```yaml
processing:
  n_workers: 4  # 1-8 typically optimal
```
- 4 threads: Good balance for most systems
- 8 threads: For high-end CPUs
- 2 threads: For limited resources

### Memory Management
Thread-based processing avoids the memory issues of multiprocessing:
- **No data duplication** between processes
- **Shared memory** for raster arrays
- **Lower overhead** than process spawning

## Validation & Quality Control

The system automatically checks for:
- ✅ Invalid NDVI values (|NDVI| > 1.0)
- ✅ Low sample sizes (< 10 pixels)
- ✅ Extreme changes (suspicious data)
- ✅ High uncertainty hexagons

Warnings are saved in metadata and printed during processing.

## Troubleshooting

### "No files found" during dataset preparation
- Check SAFE directory paths are correct
- Verify directory structure (some downloads may differ)
- Ensure you have `.jp2` or `.tif` files

### "Memory error" during processing
- Reduce H3 resolution (use 7 or 8 instead of 9)
- Decrease `n_workers` to 2
- Close other applications

### "No overlapping tiles"
- Verify GeoJSON study area is correct
- Check CRS of study area (should be EPSG:4326)
- Ensure SAFE data covers your study area

## Citation

If you use this tool in research, please cite:
```
NDVI Change Detection with H3 Hexagonal Grids
Sentinel-2 imagery analysis with adaptive classification
```
### Does extracting bands for study area cause nodata?
Yes, absolutely! The nodata is expected and correct. Here's why:Your study area (GeoJSON) is being extracted from a rectangular Sentinel-2 tile (T43SCT). The tile covers a much larger area than the area obtain, so when you clip to your polygon boundary.

## License

MIT License - feel free to use and modify.