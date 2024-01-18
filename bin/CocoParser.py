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
    original_data = coco_parser.parse_annotation_file(cat_names=["Person"], threshold=70)
    # data_with_occlusion_keypoints = coco_parser.generate_occluded_keypoints("upper_body", 0.8, 5, False)

    headers = ["img_id", "pedestrian_id", "keypoints", "target"]
    save_to_csv("train_data_original.csv", headers, original_data)
    # save_to_csv("train_data_with_occlusion.csv", headers, data_with_occlusion)

    train_file = os.path.join("..",
                              "Coco",
                              "annotations_trainval2017",
                              "person_keypoints_val2017.json")
    coco_parser = ImageProcessor(train_file)
    original_data = coco_parser.parse_annotation_file(cat_names=["Person"], threshold=70)

    headers = ["img_id", "pedestrian_id", "keypoints", "target"]
    save_to_csv("test_data.csv", headers, original_data)


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
