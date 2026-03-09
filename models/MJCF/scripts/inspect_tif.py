import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load the GeoTIFF file
tif_path = "/home/suleiman/Downloads/bc_083e021_xli1m_utm11_2019.tif"
with rasterio.open(tif_path) as src:
    elevation_data = src.read(1)  # Read first band
    profile = src.profile
    
    print(f"Shape: {elevation_data.shape}")
    print(f"Data type: {elevation_data.dtype}")
    print(f"Min elevation: {np.nanmin(elevation_data)}")
    print(f"Max elevation: {np.nanmax(elevation_data)}")
    print(f"CRS: {src.crs}")

# Visualize the raw data
# plt.imshow(elevation_data, cmap='terrain')
# plt.colorbar(label='Elevation (m)')
# plt.title('Raw LIDAR Elevation Data')
# plt.show()

def process_heightmap(elevation_data, target_size=256, smooth=True):
    """
    Process raw LIDAR data for MuJoCo heightfield
    
    Args:
        elevation_data: Raw elevation array from TIF
        target_size: Desired heightfield resolution (powers of 2 recommended)
        smooth: Apply Gaussian smoothing
    
    Returns:
        Processed heightfield as float32 array
    """
    
    # Handle no-data values (typically -9999 or NaN)
    data = elevation_data.copy().astype(float)
    data[data < -1000] = np.nanmedian(data[data > -1000])  # Replace no-data
    data = np.nan_to_num(data, nan=np.nanmedian(data))
    
    # Downsample if necessary (for performance)
    if data.shape[0] > target_size or data.shape[1] > target_size:
        from scipy.ndimage import zoom
        scale_factor = target_size / max(data.shape)
        data = zoom(data, scale_factor, order=1)
    
    # Optional: Apply smoothing to reduce noise
    if smooth:
        from scipy.ndimage import gaussian_filter
        data = gaussian_filter(data, sigma=1.0)
    
    # Normalize to reasonable range for MuJoCo (e.g., 0-10 meters)
    data_min, data_max = np.min(data), np.max(data)
    height_range = 10.0  # Adjust based on your needs
    normalized = (data - data_min) / (data_max - data_min) * height_range
    
    return normalized.astype(np.float32)

# Process your data
heightfield = process_heightmap(elevation_data, target_size=256)
print(f"Processed heightfield shape: {heightfield.shape}")
print(f"Processed height range: {heightfield.min():.2f} to {heightfield.max():.2f} m")

# MuJoCo expects heightfield as raw binary or can read from PNG
# Option A: Save as binary (raw float32)
heightfield.tofile("forest_heightfield.raw")

# Option B: Save as PNG (8-bit grayscale) - requires scaling to 0-255
heightfield_png = ((heightfield / heightfield.max()) * 255).astype(np.uint8)
from PIL import Image
Image.fromarray(heightfield_png).save("heightfield_newest.png")

# Option C: Save as NPZ for later use
np.savez_compressed("forest_heightfield.npz", heightfield=heightfield)