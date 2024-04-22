import csv
from pycocotools.coco import COCO
import os

from src.ImageParser.ImageProcessor import *

def process_crop_csv_data(annotation_file, csv_path, output_folder):
    """
    Processes a CSV file to crop images based on annotations and saves the cropped images to a specified output folder.

    Args:
        annotation_file (str): Path to the annotation file.
        csv_path (str): Path to the CSV file with 'img_id' and 'pedestrian_id' columns.
        output_folder (str): Path where the cropped images will be saved.

    The function reads the CSV, crops images for each entry using their bounding boxes, and saves them in the output folder.
    """
    coco = COCO(annotation_file)
    coco_cropper = ImageProcessor()

    with open(csv_path, mode='r') as csv_file:
        line_count = 0
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            line_count +=1
            if line_count >= 32263:
                img_id = int(row['img_id'])
                pedestrian_id = row['pedestrian_id']
                image = coco.loadImgs(img_id)[0]
                img_data = io.imread(image['coco_url'])
                anns = coco.loadAnns(int(pedestrian_id))
                for ann in anns:
                    bbox = ann['bbox']
                    cropped_image = coco_cropper.crop_image(img_data, bbox)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    cropped_image.save(os.path.join(output_folder, f"{img_id}_{pedestrian_id}.jpg"))

def main():
    data_folder = 'data'
    annotation_train_file = os.path.join(os.getcwd(),
                            "Coco",
                            "annotations_trainval2017",
                            "person_keypoints_train2017.json")

    annotation_val_file = os.path.join(os.getcwd(),
                            "Coco",
                            "annotations_trainval2017",
                            "person_keypoints_val2017.json")

    train_csv_path = os.path.join(data_folder, 'coco_train_data.csv')
    validation_csv_path = os.path.join(data_folder, 'coco_validation_data_with_occlusion.csv')
    train_output_folder = os.path.join(data_folder, 'images', 'train')
    validation_output_folder = os.path.join(data_folder, 'images', 'validation')

    process_crop_csv_data(annotation_train_file, train_csv_path, train_output_folder)
    process_crop_csv_data(annotation_val_file, validation_csv_path, validation_output_folder)

if __name__ == "__main__":
    main()