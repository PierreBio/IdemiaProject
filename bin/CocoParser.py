import os

from src.ImageParser.ImagerProcessor import ImageProcessor
from src.Common.utils import *


def main():
    file = os.path.join("..",
                        "Coco",
                        "annotations_trainval2017",
                        "person_keypoints_val2017.json")
    coco_parser = ImageProcessor(file)
    results = coco_parser.parse_annotation_file(cat_names=["Person"], threshold=70)
    results2 = coco_parser.augment_with_occlusion(weight="upper_body")

    headers = ["Img_Id", "Pedestrian_id", "Keypoints", "Target"]
    save_to_csv("output.csv", headers, results)
    save_to_csv("output2.csv", headers, results2)


def startup_msg():
    print("Starting COCO Parser...")


# TODO : occultation, data augmentation (more kps bottom/ more kps top)
# TODO : check for deep learning models (pytorch)
if __name__ == "__main__":
    startup_msg()
    main()
