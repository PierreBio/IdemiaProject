import os

from src.ImageParser.ImageProcessor import ImageProcessor
from src.Common.utils import *


def main():
    # CSV Setup
    headers = ["img_id", "pedestrian_id", "bbox", "keypoints", "target"]

    # Training
    train_file = os.path.join(os.getcwd(),
                              "Coco",
                              "annotations_trainval2017",
                              "person_keypoints_train2017.json")
    coco_parser = ImageProcessor()
    train_data = coco_parser.parse_annotation_file(train_file, cat_names=["Person"], threshold=70)

    # Saving train data
    save_to_csv(os.path.join(os.getcwd(), "data", "train_data.csv"), headers, train_data)

    # Validation
    val_file = os.path.join(os.getcwd(),
                            "Coco",
                            "annotations_trainval2017",
                            "person_keypoints_val2017.json")
    val_data = coco_parser.parse_annotation_file(val_file,
                                                 cat_names=["Person"],
                                                 threshold=70)
    occluded_data = coco_parser.apply_dynamic_occlusion_to_csv(val_data)

    # Saving validation data with & without occlusion
    save_to_csv(os.path.join(os.getcwd(), "data", "validation_data.csv"), headers, val_data)
    save_to_csv(os.path.join(os.getcwd(), "data", "validation_data_with_occlusion.csv"), headers, occluded_data)


def visualize(train_path, val_path, occl_path):
    print("TRAINING DATA")
    print("=============")
    visualize_csv_stats(train_path)

    print("\nVALIDATION DATA")
    print("=============")
    visualize_csv_stats(val_path)

    print("\nVALIDATION DATA WITH OCCLUSION")
    print("=============")
    visualize_csv_stats(occl_path)


def startup_msg():
    print("Starting COCO Parser...")


if __name__ == "__main__":
    startup_msg()
    main()
    visualize(os.path.join(os.getcwd(), "data", "train_data.csv"),
              os.path.join(os.getcwd(), "data", "validation_data.csv"),
              os.path.join(os.getcwd(), "data", "validation_data_with_occlusion.csv"))
