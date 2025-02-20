import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from affine import Affine
from shapely.geometry import shape

# Load GeoJSON files and ensure same CRS
tiles = gpd.read_file("MZKBLATT5000OGD.json")
borders = gpd.read_file("LANDESGRENZEOGD.json")
tiles = tiles.to_crs(borders.crs)  # Ensure matching CRS

in_folder = "2023/images_masks/"
out_folder = "2023/cut_images_masks/"

# Create city area from borders (assuming borders are polygons)
city_area = borders.geometry.union_all()

for idx, row in tiles.iterrows():
    # Load the corresponding image
    bnr = row["BNR5000"]
    filename = (
        f"b_s_{bnr.replace('/', '_')}_op_2023.jpg"  # Adjust filename format as needed
    )
    try:
        image = Image.open(in_folder + filename)
    except FileNotFoundError:
        print(f"Image {filename} not found, skipping.")
        continue

    image_np = np.array(image)
    tile_geom = row.geometry
    minx, miny, maxx, maxy = tile_geom.bounds

    # Calculate intersection with city area
    intersection = tile_geom.intersection(city_area)

    if intersection.is_empty:
        masked_image = np.zeros_like(image_np)
    else:
        # Create affine transformation matrix
        height, width = image_np.shape[:2]
        transform = Affine.translation(minx, maxy) * Affine.scale(
            (maxx - minx) / width, -(maxy - miny) / height
        )

        # Rasterize the intersection geometry
        mask = rasterize(
            [(intersection, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        # Apply mask to the image
        masked_image = image_np.copy()
        if masked_image.ndim == 3:  # Color image
            masked_image[mask == 0] = 0
        else:  # Grayscale image
            masked_image = masked_image * mask

    # Save the masked image
    output_filename = f"masked_{bnr.replace('/', '_')}_op_2023.jpg"
    Image.fromarray(masked_image).save(out_folder + output_filename)
