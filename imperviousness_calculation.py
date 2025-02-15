import argparse
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
import rasterio
from rasterio.features import geometry_mask
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename="calc.log", encoding="utf-8", level=logging.INFO)


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
    return parser.parse_args()


def main():
    args = get_args()
    masks = args.masks
    output = args.output
    max_width = args.max_width

    for mask_name in os.listdir(masks):
        mask_path = os.path.join(masks, mask_name)
        differences = create_geometries(mask_path, max_width)
        logger.info(f"Imperviousness for: {mask_name}")
        mask, raster_data = calculate_imperviousness(mask_path, differences)
        figure = create_figure(mask, raster_data)

        output_path = os.path.join(output, mask_name)
        cv2.imwrite(output_path, figure)


def calculate_mean_width_sampling(polygon, num_directions=100):
    centroid = polygon.centroid  # Get the polygon's centroid
    bounds = polygon.bounds
    max_radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2

    directions = np.linspace(0, 2 * np.pi, num_directions)
    widths = []

    for angle in directions:
        # Create a line extending outward from the centroid
        dx = np.cos(angle) * max_radius
        dy = np.sin(angle) * max_radius
        line = LineString(
            [(centroid.x, centroid.y), (centroid.x + dx, centroid.y + dy)]
        )

        # Intersect the line with the polygon
        intersection = polygon.intersection(line)

        if intersection.is_empty:
            continue

        # Calculate the distance (width) for this direction
        if intersection.geom_type == "MultiLineString":
            # Sum lengths if multiple intersections
            width = sum(part.length for part in intersection.geoms)
        else:
            width = intersection.length

        widths.append(width)

    # Calculate the mean width
    mean_width = np.mean(widths) if widths else 0
    return mean_width


# Function to calculate buffer distance for a given ratio
def calculate_buffer_distance(geometry, ratio, max_width):
    original_width = calculate_mean_width_sampling(geometry)
    # 3rd quadrant width of tree crowns
    if original_width > max_width:
        original_width = max_width

    scaled_width = original_width * ratio
    return (scaled_width - original_width) / 2


def create_geometries(image_path, max_width):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = np.array(image)

    # Step 2: Define the target BGR color (from the hex value #226126)
    lower_dark_green = np.array([0, 50, 0])
    upper_dark_green = np.array([100, 150, 100])

    # Step 3: Create a binary mask for the target color
    binary_mask = cv2.inRange(image_array, lower_dark_green, upper_dark_green)

    # Step 4: Extract contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 5: Convert contours to Shapely geometries
    differences = []

    ratio = 2.24

    for contour in contours:
        # Approximate contour as a polygon
        coords = contour.squeeze().tolist()  # Extract x, y coordinates
        if len(coords) >= 3:  # Only consider valid polygons
            geom = Polygon(coords)

            distance = calculate_buffer_distance(geom, ratio, max_width)
            buffered_geom = geom.buffer(distance)
            difference = buffered_geom.difference(geom)

            differences.append(difference)

    return differences


def calculate_imperviousness(image_path, differences):
    with rasterio.open(image_path) as src:
        raster_data = src.read((1, 2, 3))
        transform = src.transform

        # Create a mask for the difference geometries
        mask = geometry_mask(
            [geom.__geo_interface__ for geom in differences],
            transform=transform,
            invert=True,
            out_shape=(src.height, src.width),
        )

        # Apply the mask to the raster data
        masked_data = raster_data[:, mask]

    # Define the color codes of interest in their corresponding raster values (BGR)
    color_mapping = {
        "#FFFFFF": (255, 255, 255),  # mapping for white
        "#DE1F07": (222, 31, 7),  # mapping for red
        "#949494": (148, 148, 148),  # mapping for gray
    }

    # Count pixels for each color
    total_pixels = masked_data.shape[1]
    color_counts = {}

    for color, bgr in color_mapping.items():
        # Count pixels where all bands match the target BGR
        color_counts[color] = np.sum(
            (masked_data[0] == bgr[0])
            & (masked_data[1] == bgr[1])
            & (masked_data[2] == bgr[2])
        )

    # Calculate percentages
    percentages = {
        color: (count / total_pixels) * 100 for color, count in color_counts.items()
    }
    percentages["other"] = 100 - sum(percentages.values())

    logger.info("Color contribution for area near trees")
    for color, percentage in percentages.items():
        logger.info(f"{color}: {percentage:.2f} percent")

    image_pixels = src.height * src.width

    difference_other_pixels = total_pixels - sum(color_counts.values())

    contribution_percentages = {
        color: (count / image_pixels) * 100 for color, count in color_counts.items()
    }
    contribution_percentages["other"] = (difference_other_pixels / image_pixels) * 100

    logger.info("Color contribution for total area")
    for color, percentage in contribution_percentages.items():
        logger.info(f"{color}: {percentage:.2f}% of the total image area")

    total_imp = (
        sum(contribution_percentages.values()) - contribution_percentages["other"]
    )
    logger.info(f"Total imperviousness: {total_imp}")

    return mask, raster_data


def create_figure(mask, raster_data):
    bgr_masked = np.zeros_like(raster_data)

    for i in range(3):  # Loop through B, G, R bands
        band = raster_data[i]  # Extract each band
        bgr_masked[i][mask] = band[mask]  # Apply the mask

    # Transpose to (height, width, 3) for visualization
    return np.transpose(bgr_masked, (1, 2, 0))


if __name__ == "__main__":
    main()
