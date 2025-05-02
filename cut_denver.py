import os
import glob
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask

# Paths
shapefile_path = "CBD.shp"
tif_folder = os.path.dirname(shapefile_path)
tif_files = glob.glob(os.path.join(tif_folder, "*.tif"))

# Load shapefile
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(epsg=4326)  # Ensure common CRS

# Get geometry
shapes = [feature["geometry"] for feature in gdf.__geo_interface__["features"]]

# Merge TIFFs
src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]
mosaic, out_trans = merge(src_files_to_mosaic)

# Copy metadata
out_meta = src_files_to_mosaic[0].meta.copy()
out_meta.update(
    {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": src_files_to_mosaic[0].crs,
    }
)

# Save merged image
merged_path = os.path.join(tif_folder, "merged.tif")
with rasterio.open(merged_path, "w", **out_meta) as dest:
    dest.write(mosaic)

# Clip with shapefile
with rasterio.open(merged_path) as src:
    clipped_image, clipped_transform = mask(src, shapes, crop=True)
    clipped_meta = src.meta.copy()
    clipped_meta.update(
        {
            "height": clipped_image.shape[1],
            "width": clipped_image.shape[2],
            "transform": clipped_transform,
        }
    )

# Save clipped image
clipped_path = os.path.join(tif_folder, "clipped.tif")
with rasterio.open(clipped_path, "w", **clipped_meta) as dest:
    dest.write(clipped_image)

print(f"Merged image saved to: {merged_path}")
print(f"Clipped image saved to: {clipped_path}")
