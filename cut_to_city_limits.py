import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from affine import Affine

# Load GeoJSON files and ensure same CRS
tiles = gpd.read_file("MZKBLATT5000OGD.json")
borders = gpd.read_file("LANDESGRENZEOGD.json")
tiles = tiles.to_crs(borders.crs)

in_folder = "2023/images_masks/"
out_folder = "2023/cut_images_masks/"

city_area = borders.geometry.unary_union

for idx, row in tiles.iterrows():
    bnr = row["BNR5000"]
    filename = f"b_s_{bnr.replace('/', '_')}_op_2023.jpg"
    try:
        image = Image.open(in_folder + filename)
    except FileNotFoundError:
        print(f"Image {filename} not found, skipping.")
        continue

    image_np = np.array(image)
    tile_geom = row.geometry
    minx, miny, maxx, maxy = tile_geom.bounds

    intersection = tile_geom.intersection(city_area)

    if intersection.is_empty:
        masked_image = np.zeros_like(image_np)
    else:
        # Create transformation matrix
        height, width = image_np.shape[:2]
        transform = Affine.translation(minx, maxy) * Affine.scale(
            (maxx - minx) / width, -(maxy - miny) / height
        )

        mask = rasterize(
            [(intersection, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        masked_image = image_np.copy()
        masked_image[mask == 0] = 0

    output_filename = f"masked_{bnr.replace('/', '_')}_op_2023.jpg"
    Image.fromarray(masked_image).save(out_folder + output_filename)
