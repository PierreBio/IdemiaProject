import os

from src.ImageParser.ImagerProcessor import ImageProcessor
from src.Common.utils import *


def main():
    # file = os.path.join("..",
    #                     "Coco",
    #                     "annotations_trainval2017",
    #                     "person_keypoints_val2017.json")
    # coco_parser = ImageProcessor(file)
    # original_data = coco_parser.parse_annotation_file(cat_names=["Person"], threshold=70)
    # occluded_only = coco_parser.generate_occluded_data("upper_body", False)
    #
    # headers = ["Img_Id", "Pedestrian_id", "Keypoints", "Target"]
    # save_to_csv("orignal_data.csv", headers, original_data)
    # save_to_csv("occluded_only.csv", headers, occluded_only)
    visualize_csv_stats("occluded_only.csv")


def startup_msg():
    print("Starting COCO Parser...")


# TODO : occultation, data augmentation (more kps bottom/ more kps top)
# TODO : check for deep learning models (pytorch)
if __name__ == "__main__":
    # startup_msg()
    main()
