from pycocotools.coco import COCO
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import json
import csv


def main(data_path):
    saved_data = []
    if os.path.isfile(data_path):
        # Get all images with "persons" in it
        coco = COCO(data_path)
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)
        for img_id in img_ids:
            img = coco.loadImgs(img_id)[0]
            img_anns = coco.loadAnns(coco.getAnnIds(img_id))

            for img_ann in img_anns:
                img_kps = img_ann["keypoints"]

                # Checking if feets are visible (last & 3rd last elements of keypoints)
                if img_kps[-1] != 0 and img_kps[-4] != 0:
                    if are_keypoints_visible(img_kps[:-6], 70):
                        print("accepted")
                        # display_img(coco, img, img_anns)
                        target = [(img_kps[-3] + img_kps[-6]) / 2, (img_kps[-3] + img_kps[-6]) / 2]
                        saved_data.append(prepare_data(img_id, img_kps[:-6], target))
                        break
                    else:
                        print(f"REJECTED: Not enough keypoints visble image {img_id}")
                else:
                    print(f"REJECTED: Feets no visible for image {img_id}")
                    # display_img(coco, img, img_anns)

            save_to_csv("output", saved_data)
    else:
        print(f"File {ann_file_path} does not exist, aborting...")
        exit(-1)


def display_img(coco_db, image, annotations):
    # Display img from url
    image = io.imread(image["coco_url"])

    plt.imshow(image)
    # Display kps
    coco_db.showAnns(annotations)

    plt.axis('off')
    plt.show()


def are_keypoints_visible(keypoint_list, percentage_threshold):
    # Extracting visibility values from keypoints
    v_values = keypoint_list[2::3]

    # Count the number of non-zero visibility values
    visible_count = sum(v != 0 for v in v_values)

    # Check if number of visible keypoints is above threshold
    total_keypoints = len(v_values)
    visible_percentage = (visible_count / total_keypoints) * 100

    # Check if the visible percentage is above the threshold
    return visible_percentage >= percentage_threshold


def prepare_data(*args):
    return [arg for arg in args]


def save_to_csv(file_name, data_list):
    # Define file  headers
    headers = ["Img_Id", "Keypoints", "Target"]

    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the headers
        writer.writerow(headers)

        # Write the data
        for row in data_list:
            writer.writerow(row if isinstance(row, list) else [row])


def startup_msg():
    print("Starting COCO Parser...")


if __name__ == "__main__":
    startup_msg()
    ann_file_path = os.path.join("../data/annotations_trainval2017/annotations/person_keypoints_val2017.json")
    main(ann_file_path)
