import os

from src.ImageParser.ImageProcessor import ImageProcessor
from src.Common.utils import *


def main():
    # CSV Setup
    headers = ["img_id", "pedestrian_id", "bbox", "keypoints", "target"]

    # Training/ validation ?
    train_file = os.path.join(os.getcwd(),
                              "OCHuman",
                              "ochuman_coco_format_val_range_0.00_1.00.json")
    coco_parser = ImageProcessor()
    train_data = coco_parser.parse_annotation_file(train_file, cat_names=["Person"], threshold=70)

    # Saving train data
    save_to_csv(os.path.join(os.getcwd(), "data",
                "och_train_data.csv"), headers, train_data)

    # Validation, test ?
    val_file = os.path.join(os.getcwd(), "OCHuman", "ochuman_coco_format_test_range_0.00_1.00.json")
    coco_parser = ImageProcessor()
    val_data = coco_parser.parse_annotation_file(val_file, cat_names=["Person"],threshold=70)

    # Saving Validation data
    save_to_csv(os.path.join(os.getcwd(), "data",
                "och_validation_data.csv"), headers, val_data)


def visualize(train_path, val_path):
    print("TRAINING DATA")
    print("=============")
    visualize_csv_stats(train_path)

    print("\nVALIDATION DATA")
    print("=============")
    visualize_csv_stats(val_path)


def startup_msg():
    print("Starting OCHuman Parser...")


if __name__ == "__main__":
    startup_msg()
    main()
    visualize(os.path.join(os.getcwd(), "data", "train_data_OCH.csv"),
              os.path.join(os.getcwd(), "data", "validation_data_OCH.csv"))
