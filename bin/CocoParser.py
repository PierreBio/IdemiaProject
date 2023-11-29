from pycocotools.coco import COCO
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import json


def main():
    # TODO: make path dynamic
    json_output = []
    ann_file_path = os.path.join("../Coco/annotations_trainval2017/person_keypoints_val2017.json")
    if os.path.isfile(ann_file_path):
        # Get all images with "persons" in it
        coco = COCO(ann_file_path)
        cat_ids = coco.getCatIds(catNms=['person'])
        img_ids = coco.getImgIds(catIds=cat_ids)

        for img_id in img_ids:
            img = coco.loadImgs(img_id)[0]

            img_anns = coco.loadAnns(coco.getAnnIds(img_id))

            for img_ann in img_anns:
                img_kps = img_ann["keypoints"]

                # Checking if feets are visible (6 last elements of keypoints)
                # TODO: x, y , visibility (check only visibility == 1 & == 2)
                if all(keypoint != 0 for keypoint in img_kps[-6:]):
                    print("accepted")
                    display_img(coco, img, img_ann)
                    # json_output.append({"img_id": img["coco_url"]})
                    break
                else:
                    print("rejected")
                    display_img(coco, img, img_ann)

    else:
        print(f"File {ann_file_path} does not exist, aborting...")
        exit(-1)


def display_img(coco_db, image, annotations):
    # Display img from url
    image = io.imread(image["coco_url"])

    # Display kps
    coco_db.showAnns(annotations)

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def save_img_url():
    with open("custom_db.json") as f:
        json.dump


def startup_msg():
    print("Starting COCO Parser...")


if __name__ == "__main__":
    startup_msg()
    main()
