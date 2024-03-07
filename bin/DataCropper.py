import csv
from pycocotools.coco import COCO
import os

from src.ImageParser.ImageProcessor import *

def process_crop_csv_data(annotation_file, csv_path, output_folder):
    coco = COCO(annotation_file)
    coco_cropper = ImageProcessor()

    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            img_id = int(row['img_id'])
            pedestrian_id = row['pedestrian_id']

            image = coco.loadImgs(img_id)[0]
            img_data = io.imread(image['coco_url'])

            anns = coco.loadAnns(int(pedestrian_id))
            for ann in anns:
                bbox = ann['bbox']
                cropped_filename = f"{img_id}_{pedestrian_id}.jpg"
                coco_cropper.crop_and_save_image(img_data, bbox, output_folder, cropped_filename)

def main():
    data_folder = 'data'
    annotation_file = os.path.join(os.getcwd(),
                            "Coco",
                            "annotations_trainval2017",
                            "person_keypoints_train2017.json")

    train_csv_path = os.path.join(data_folder, 'coco_train_data.csv')
    validation_csv_path = os.path.join(data_folder, 'coco_validation_data_with_occlusion.csv')
    train_output_folder = os.path.join(data_folder, 'images', 'train')
    validation_output_folder = os.path.join(data_folder, 'images', 'validation')

    process_crop_csv_data(annotation_file, train_csv_path, train_output_folder)
    process_crop_csv_data(annotation_file, validation_csv_path, validation_output_folder)

if __name__ == "__main__":
    main()