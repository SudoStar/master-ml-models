import cv2
import numpy as np
import os


def combine_images_cv2(image_path, output_path):
    """Combines 9 images into a 3x3 grid using OpenCV.

    Args:
        image_paths: A list of 9 image paths.
        output_path: The path to save the combined image.
    """

    images = []
    widths = []
    heights = []

    for filename in os.listdir(image_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            path = os.path.join(image_path, filename)
            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"Could not read image: {path}")
                images.append(img)
                height, width, _ = img.shape
                widths.append(width)
                heights.append(height)
            except FileNotFoundError:
                raise FileNotFoundError(f"Image not found: {path}")
            except ValueError as e:
                raise e
            except Exception as e:
                raise Exception(f"Error processing image {path}: {e}")

    # Ensure all images have the same dimensions
    if not all(w == widths[0] for w in widths) or not all(
        h == heights[0] for h in heights
    ):
        # Resize images
        new_width = min(widths)
        new_height = min(heights)
        resized_images = []
        for image in images:
            resized_image = cv2.resize(image, (new_width, new_height))
            resized_images.append(resized_image)
        images = resized_images
        print(
            "Images were resized to be the same size for a clean grid. Consider using images of the same dimensions for best quality."
        )

    grid_width = 3 * images[0].shape[1]
    grid_height = 3 * images[0].shape[0]
    combined_image = np.zeros(
        (grid_height, grid_width, 3), dtype=np.uint8
    )  # Create an empty image (3 channels for color)

    image_index = 0
    for row in range(3):
        for col in range(3):
            x_offset = col * images[0].shape[1]
            y_offset = row * images[0].shape[0]

            combined_image[
                y_offset : y_offset + images[0].shape[0],
                x_offset : x_offset + images[0].shape[1],
            ] = images[image_index]
            image_index += 1

    cv2.imwrite(output_path, combined_image)
    print(f"Combined image saved to: {output_path}")


image_path = "poc_images"
output_path = "large_mask/combined_image.jpg"

combine_images_cv2(image_path, output_path)
