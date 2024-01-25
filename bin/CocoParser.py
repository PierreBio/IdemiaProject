import os

from src.ImageParser.ImagerProcessor import ImageProcessor
from src.Common.utils import *


def main():
    # Training
    train_file = os.path.join("..",
                              "Coco",
                              "annotations_trainval2017",
                              "person_keypoints_train2017.json")
    coco_parser = ImageProcessor(train_file)
    train_data = coco_parser.parse_annotation_file(cat_names=["Person"], threshold=70)

    headers = ["img_id", "pedestrian_id", "bbox", "keypoints", "target"]
    save_to_csv("train_data.csv", headers, train_data)

    # Validation
    val_file = os.path.join("..",
                            "Coco",
                            "annotations_trainval2017",
                            "person_keypoints_val2017.json")
    coco_parser = ImageProcessor(val_file)
    val_data = coco_parser.parse_annotation_file(cat_names=["Person"], threshold=70)

    headers = ["img_id", "pedestrian_id", "bbox", "keypoints", "target"]
    save_to_csv("validation_data.csv", headers, val_data)


def visualize():
    print("ORIGINAL DATA")
    print("=============")
    visualize_csv_stats("train_data_original.csv")

    print("\nOCCLUDED DATA")
    print("=============")
    visualize_csv_stats("train_data_with_occlusion.csv")


def startup_msg():
    print("Starting COCO Parser...")


if __name__ == "__main__":
    startup_msg()
    main()
    visualize()
