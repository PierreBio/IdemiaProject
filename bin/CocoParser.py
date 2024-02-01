import os

from src.ImageParser.ImageProcessor import ImageProcessor
from src.Common.utils import *


def main():
    # CSV Setup
    headers = ["img_id", "pedestrian_id", "bbox", "keypoints", "target"]

    # Training
    train_file = os.path.join("..",
                              "Coco",
                              "annotations_trainval2017",
                              "person_keypoints_train2017.json")
    coco_parser = ImageProcessor(train_file)
    train_data = coco_parser.parse_annotation_file(cat_names=["Person"], threshold=70)

    # Saving train data
    save_to_csv("../data/train_data.csv", headers, train_data)

    # Validation
    val_file = os.path.join("..",
                            "Coco",
                            "annotations_trainval2017",
                            "person_keypoints_val2017.json")
    coco_parser = ImageProcessor(val_file)
    val_data = coco_parser.parse_annotation_file(cat_names=["Person"],
                                                 threshold=70)
    augmented_val_data_box = coco_parser.generate_occluded_box(occlusion_chance=0.8,
                                                               range_occlusion=(0.5, 1),
                                                               include_original_data=False)
    augmented_val_data_kps = coco_parser.generate_occluded_keypoints(weight_position="",
                                                                     weight_value=0.7,
                                                                     min_visible_threshold=5,
                                                                     include_original_data=False)
    combined_val_data = augmented_val_data_box + augmented_val_data_kps

    # Saving Validation data
    save_to_csv("../data/validation_data.csv", headers, combined_val_data)


def visualize(train_path, val_path):
    print("TRAINING DATA")
    print("=============")
    visualize_csv_stats(train_path)

    print("\nVALIDATION DATA")
    print("=============")
    visualize_csv_stats(val_path)


def startup_msg():
    print("Starting COCO Parser...")


if __name__ == "__main__":
    startup_msg()
    main()
    visualize("../data/train_data.csv",
              "../data/validation_data.csv")
