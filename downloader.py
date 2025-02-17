import urllib.error
import urllib.request
import argparse
import os
import zipfile
import cv2


# vienna_2014_url_example = "https://www.wien.gv.at/ma41datenviewer/downloads/geodaten/op_img/34_4_op_2024.zip"

url = "https://www.wien.gv.at/ma41datenviewer/downloads/geodaten/op_img/"


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "-s",
        "--start",
        type=int,
        default=12,
        help="Start image tile",
    )
    arg("-e", "--end", type=int, default=59, help="End image tile")
    arg(
        "-yf",
        "--year_from",
        type=str,
        default="op2014",
        help="First year to get images for",
    )
    arg(
        "-yt",
        "--year_to",
        type=str,
        default="op_2024",
        help="Second year to get images for",
    )
    arg("-sr", "--scale_ratio", help="image scale ratio", type=float, default=0.25)
    return parser.parse_args()


year_from_folder = "from/"
year_to_folder = "to/"


def main():
    args = get_args()

    start = args.start
    end = args.end
    year_from = args.year_from
    year_to = args.year_to
    scale_ratio = args.scale_ratio

    download_images(start, end, year_from, year_to)

    extract_images(year_from_folder)
    extract_images(year_to_folder)

    downscale_images(year_from_folder, scale_ratio)
    downscale_images(year_to_folder, scale_ratio)


def downscale_images(directory, scale_ratio):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):

            img_path = os.path.join(directory, filename)

            img = cv2.imread(img_path)
            img = cv2.resize(
                img,
                None,
                fx=scale_ratio,
                fy=scale_ratio,
                interpolation=cv2.INTER_AREA,
            )

            new_path = os.path.join(directory, "s_" + filename)
            cv2.imwrite(new_path, img)

    print("All JPEG images have been downscaled by 75 percent and saved.")


def extract_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            # Define the full path to the ZIP file
            zip_path = os.path.join(directory, filename)

            # Open the ZIP file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Extract all files
                zip_ref.extractall(directory)

            # Delete the ZIP file
            os.remove(zip_path)

    for filename in os.listdir(directory):
        # Check if the file is not a JPEG and delete it
        if not filename.endswith(".jpg") and not filename.endswith(".jpeg"):
            # Define the full path to the file
            file_path = os.path.join(directory, filename)

            # Delete the file
            os.remove(file_path)


def download_images_for_existing_images(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(directory, filename)

            zip_name = os.path.splitext(img_path)[0] + ".zip"
            zip_name = zip_name.replace("2024", "2023")
            url_zip = url + zip_name

            urllib.request.urlretrieve(url=url_zip, filename=("2023" + zip_name))
            print(f"Downloaded {zip_name}")


def download_images(start, end, year_from, year_to):
    for i in range(start, end + 1):
        part_one = str(i) + "_"
        for j in range(1, 5):
            part_two = str(j) + "_"
            filename_from = part_one + part_two + year_from + ".zip"
            filename_to = part_one + part_two + year_to + ".zip"

            url_from = url + filename_from
            url_to = url + filename_to

            try:
                urllib.request.urlopen(url_from)
            except urllib.error.URLError as e:
                print(f"{url_from} not found", e)
                continue

            try:
                urllib.request.urlopen(url_to)
            except urllib.error.URLError as e:
                print(f"{url_to} not found", e)
                continue

            urllib.request.urlretrieve(
                url=url_from, filename=(year_from_folder + filename_from)
            )
            print(f"Downloaded {filename_from}")

            urllib.request.urlretrieve(
                url=url_to, filename=(year_to_folder + filename_to)
            )
            print(f"Downloaded {filename_to}")


if __name__ == "__main__":
    main()
