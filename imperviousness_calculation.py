import argparse
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
import rasterio
from rasterio.features import geometry_mask
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="calc.log",
    level=logging.INFO,
    format="%(asctime)s :: %(message)s",
)

# 1.75
# 2.24
# 2.8


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-m",
        "--masks",
        type=str,
        default="masks",
        help="Masks folder",
    )
    arg("-o", "--output", type=str, default="output", help="Output folder")
    arg(
        "--max_width", type=int, default=8, help="Max buffer distance calculation width"
    )
    arg(
        "-r",
        "--ratio",
        type=float,
        default=2.24,
        help="Root spread to crown spread ratio",
    )
    return parser.parse_args()


def main():
    args = get_args()
    masks = args.masks
    output = args.output
    max_width = args.max_width
    ratio = args.ratio

    imp_results = []
    tree_areas = []

    for mask_name in os.listdir(masks):
        if mask_name.endswith(".jpg") or mask_name.endswith(".jpeg"):
            mask_path = os.path.join(masks, mask_name)
            geoms, differences = create_geometries(mask_path, max_width, ratio)
            tree_area = sum(p.area for p in geoms)
            tree_areas.append(tree_area)

            logger.info(f"Imperviousness for: {mask_name} with ratio {str(ratio)}")
            logger.info(f"Tree cluster area: {tree_area}")

            mask, raster_data, imp = calculate_imperviousness(mask_path, differences)
            imp_results.append(imp)

            if output == "none":
                continue

            figure = create_figure(mask, raster_data)
            output_path = os.path.join(output, mask_name)
            cv2.imwrite(output_path, figure)

    avg_imp = sum(imp_results) / len(imp_results)
    total_area = sum(tree_areas)

    logger.info(f"Total tree cluster area: {total_area}")
    logger.info(f"Average imperviousness near trees: {avg_imp}")


def calculate_mean_width_sampling(polygon, num_directions=100):
    centroid = polygon.centroid
    bounds = polygon.bounds
    max_radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2

    directions = np.linspace(0, 2 * np.pi, num_directions)
    widths = []

    for angle in directions:
        dx = np.cos(angle) * max_radius
        dy = np.sin(angle) * max_radius
        line = LineString(
            [(centroid.x, centroid.y), (centroid.x + dx, centroid.y + dy)]
        )

        intersection = polygon.intersection(line)

        if intersection.is_empty:
            continue

        if intersection.geom_type == "MultiLineString":
            width = sum(part.length for part in intersection.geoms)
        else:
            width = intersection.length

        widths.append(width)

    mean_width = np.mean(widths) if widths else 0
    return mean_width


def calculate_buffer_distance(geometry, ratio, max_width):
    original_width = calculate_mean_width_sampling(geometry)
    if original_width > max_width:
        original_width = max_width

    scaled_width = original_width * ratio
    return (scaled_width - original_width) / 2


def create_geometries(image_path, max_width, ratio):
    with rasterio.open(image_path) as src:
        # Read bands in BGR order (bands 3, 2, 1 for RGB images)
        bgr_image = np.dstack([src.read(3), src.read(2), src.read(1)])

    lower_dark_green = np.array([0, 50, 0])  # BGR lower bound for dark green
    upper_dark_green = np.array([100, 150, 100])  # BGR upper bound

    binary_mask = cv2.inRange(bgr_image, lower_dark_green, upper_dark_green)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    differences = []
    geoms = []

    for contour in contours:
        coords = contour.squeeze().tolist()
        if len(coords) >= 3:
            geom = Polygon(coords)
            geoms.append(geom)

            distance = calculate_buffer_distance(geom, ratio, max_width)
            buffered_geom = geom.buffer(distance)
            differences.append(buffered_geom.difference(geom))

    return geoms, differences


def calculate_imperviousness(image_path, differences):
    with rasterio.open(image_path) as src:
        # Read bands in BGR order (3, 2, 1)
        raster_data = np.stack([src.read(3), src.read(2), src.read(1)], axis=0)
        transform = src.transform

        mask = geometry_mask(
            differences,
            transform=transform,
            invert=True,
            out_shape=(src.height, src.width),
        )

        masked_data = raster_data[:, mask]

    # Define color ranges instead of exact matches
    color_ranges = {
        "#FFFFFF": (  # White
            np.array([210, 210, 210], dtype=np.uint8),  # Lower bound (B, G, R)
            np.array([255, 255, 255], dtype=np.uint8),  # Upper bound (B, G, R)
        ),
        "#DE1F07": (  # Red
            np.array([0, 60, 200], dtype=np.uint8),  # Lower bound (B, G, R)
            np.array([50, 100, 255], dtype=np.uint8),  # Upper bound (B, G, R)
        ),
        "#949494": (  # Gray
            np.array([130, 130, 130], dtype=np.uint8),  # Lower bound (B, G, R)
            np.array([160, 160, 160], dtype=np.uint8),  # Upper bound (B, G, R)
        ),
    }

    color_counts = {color: 0 for color in color_ranges}

    for color, (lower, upper) in color_ranges.items():
        # Count pixels within the specified range
        in_range = np.logical_and.reduce(
            [
                masked_data[0] >= lower[0],
                masked_data[0] <= upper[0],
                masked_data[1] >= lower[1],
                masked_data[1] <= upper[1],
                masked_data[2] >= lower[2],
                masked_data[2] <= upper[2],
            ]
        )
        color_counts[color] = np.sum(in_range)

    total_pixels = masked_data.shape[1]
    percentages = {
        color: (count / total_pixels) * 100 for color, count in color_counts.items()
    }
    percentages["total"] = sum(percentages.values())

    logger.info("Color contribution for area near trees")
    for color, percentage in percentages.items():
        logger.info(f"{color}: {percentage:.2f} percent")

    return mask, raster_data, percentages["total"]


def create_figure(mask, raster_data):
    bgr_masked = np.zeros_like(raster_data)
    for i in range(3):
        bgr_masked[i][mask] = raster_data[i][mask]
    return np.transpose(bgr_masked, (1, 2, 0))


if __name__ == "__main__":
    main()
