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
    coco_parser = ImageProcessor(train_file)
    train_data = coco_parser.parse_annotation_file(
        cat_names=["Person"], threshold=70)

    # Saving train data

    save_to_csv(os.path.join(os.getcwd(), "data",
                "train_data_OCH.csv"), headers, train_data)

    # Validation, test ?
    val_file = os.path.join(os.getcwd(),
                            "OCHuman",
                            "ochuman_coco_format_test_range_0.00_1.00.json")
    coco_parser = ImageProcessor(val_file)
    val_data = coco_parser.parse_annotation_file(cat_names=["Person"],
                                                 threshold=70)

    # augmented_val_data_box = coco_parser.generate_occluded_box(occlusion_chance=0.8,
    #                                                            range_occlusion=(
    #                                                                0.5, 1),
    #                                                            include_original_data=False)
    # augmented_val_data_kps = coco_parser.generate_occluded_keypoints(weight_position="",
    #                                                                  weight_value=0.7,
    #                                                                  min_visible_threshold=5,
    #                                                                  include_original_data=False)
    # combined_val_data = augmented_val_data_box + augmented_val_data_kps

    # Saving Validation data
    save_to_csv(os.path.join(os.getcwd(), "data",
                "validation_data_OCH.csv"), headers, val_data)


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
