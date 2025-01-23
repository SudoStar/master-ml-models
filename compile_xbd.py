import os
import argparse
import numpy as np
import rasterio
from pathlib import Path
from PIL import Image
import cv2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_OpenEarthMap",
        type=str,
        default="./OpenEarthMap",
    )
    parser.add_argument(
        "--path_to_xBD",
        type=str,
        default="./xBD",
    )
    args = parser.parse_args()

    ROOT = args.path_to_xBD
    OEMDIR = args.path_to_OpenEarthMap

    xbd_fullimgs = [
        [f.name, str(f)]
        for f in Path(ROOT).rglob("*.png")
        if f.name.endswith("pre_disaster.png")
    ]
    xbd_fullimgs = np.asarray(xbd_fullimgs)
    xbd_missing = np.loadtxt(os.path.join(OEMDIR, "xbd_files.csv"), dtype=str, delimiter=",")

    print("fullimgs: ", len(xbd_fullimgs), "missing: ", len(xbd_missing))

    for f1, f2 in xbd_missing:
        idx = np.where(xbd_fullimgs[:, 0] == f1)[0]

        dir = xbd_fullimgs[idx, 1][0]
        fnm = xbd_fullimgs[idx, 0][0]

        #if len(f2.split("_")) == 2:
        #    city = f2.split("_")[0]
        #else:
        city = "_".join(f2.split("_")[:-1])

        img_path = os.path.join(ROOT, fnm)
        oem_path = os.path.join(OEMDIR, city, "labels", f2)

        # read xBD PNG image
        # img = np.asarray(Image.open(img_path))
        img = cv2.imread(img_path)
        tif_path = oem_path.replace("/labels/", "/images/")
        cv2.imwrite(tif_path, img)

        # if os.path.exists(oem_path):
        # Read OpenEarthMap label geotiff image
        #    with rasterio.open(oem_path, "r") as src:
        #        profile = src.profile
        #    profile.update(count=3)
        #    tif_path = oem_path.replace("/labels/", "/images/")
        #    with rasterio.open(tif_path, "w", **profile) as dst:
        #        for i in range(3):
        #            dst.write(img[:, :, i], i + 1)
        #else:
            # Create a default profile if oem_path does not exist
            # profile = {
            #     'driver': 'GTiff',
            #     'dtype': 'uint8',
            #     'nodata': None,
            #     'width': img.shape[1],
            #     'height': img.shape[0],
            #     'count': 3,
            #     'crs': None,
            #     'transform': None
            # }

            # Save xBD images into OpenEarthMap as geotiff image
            # profile.update(count=3)
            #tif_path = oem_path.replace("/labels/", "/images/")
            #with rasterio.open(tif_path, "w") as dst:
            #    for i in range(3):
            #        dst.write(img[:, :, i], i + 1)
