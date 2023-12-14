import os

from src.ImageParser.ImagerProcessor import ImageProcessor
from src.Common.utils import *


def main():
    file = os.path.join("..",
                        "Coco",
                        "annotations_trainval2017",
                        "person_keypoints_val2017.json")
    coco_parser = ImageProcessor(file)
    original_data = coco_parser.parse_annotation_file(cat_names=["Person"], threshold=70)
    occluded_only = coco_parser.generate_occluded_data("upper_body", 0.8, 5, False)

    headers = ["img_id", "pedestrian_id", "keypoints", "target"]
    save_to_csv("orignal_data.csv", headers, original_data)
    save_to_csv("occluded_only_w_threshold.csv", headers, occluded_only)


def visualize():
    print("ORIGINAL DATA")
    print("=============")
    visualize_csv_stats("orignal_data.csv")

    print("\nOCCLUDED DATA")
    print("=============")
    visualize_csv_stats("occluded_only_w_threshold.csv")


def startup_msg():
    print("Starting COCO Parser...")


if __name__ == "__main__":
    startup_msg()
    # main()
    visualize()
