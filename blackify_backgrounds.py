import cv2
import numpy as np
import os


def convert_to_black_background(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(directory, filename)

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            mask = cv2.inRange(img, (250, 250, 250), (255, 255, 255))

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8, ltype=cv2.CV_32S
            )

            area_threshold = 10000  # Example threshold

            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= area_threshold:
                    component_mask = (labels == i).astype(bool)
                    img[component_mask] = (0, 0, 0)  # BGR to black

            new_path = os.path.join(directory, "b_" + filename)
            cv2.imwrite(new_path, img)


convert_to_black_background("2023/images")
