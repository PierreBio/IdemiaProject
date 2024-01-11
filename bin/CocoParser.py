import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from src.ImageParser.ImagerProcessor import ImageProcessor
from src.Common.utils import *


def main():
    file = os.path.join("..",
                        "Coco",
                        "annotations_trainval2017",
                        "person_keypoints_val2017.json")
    coco_parser = ImageProcessor(file)
    original_data = coco_parser.parse_annotation_file(cat_names=["Person"], threshold=70)
    data_with_occlusion = coco_parser.generate_occluded_data("upper_body", 0.8, 5, True)
    coco_parser.extract_and_save_image_crops("crops")

    headers = ["img_id", "pedestrian_id", "keypoints", "target"]
    save_to_csv("original_data.csv", headers, original_data)
    save_to_csv("data_with_occlusion.csv", headers, data_with_occlusion)


def visualize():
    print("ORIGINAL DATA")
    print("=============")
    visualize_csv_stats("orignal_data.csv")

    print("\nOCCLUDED DATA")
    print("=============")
    visualize_csv_stats("data_with_occlusion.csv")


def startup_msg():
    print("Starting COCO Parser...")


if __name__ == "__main__":
    startup_msg()
    main()
    visualize()
