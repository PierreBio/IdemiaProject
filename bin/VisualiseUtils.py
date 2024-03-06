from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os
import numpy as np


def visualize_specific_pedestrian_local(annotation_file, images_folder_path, image_id, pedestrian_id, pred, truth):
    # Initialize COCO api
    coco = COCO(annotation_file)

    # Load image metadata
    img = coco.loadImgs(int(image_id))[0]

    # Construct image path
    # Assuming the image filename is formatted with 12 digits
    img_filename = f"{str(image_id).zfill(12)}.jpg"
    img_path = os.path.join(images_folder_path, img_filename)

    # Load the image from the local path
    image = Image.open(img_path)

    # Display the image
    plt.imshow(image)
    plt.axis('off')

    ax = plt.gca()
    anns = coco.loadAnns(int(pedestrian_id))

    for ann in anns:
        # Draw keypoints
        if "keypoints" in ann:
            keypoints = ann["keypoints"]
            for i in range(0, len(keypoints), 3):
                kp_x, kp_y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                # Draw only if keypoint is labeled and visible
                if v == 2:
                    ax.plot(kp_x, kp_y, 'o', color='blue', markersize=3)

        x0, y0, width, height = ann["bbox"]
        denormalized_pred_x = (pred[0] * width) + x0
        denormalized_pred_y = (pred[1] * height) + y0
        denormalized_truth_x = (truth[0] * width) + x0
        denormalized_truth_y = (truth[1] * height) + y0
        ax.plot(denormalized_pred_x, denormalized_pred_y, 'o', color='red', markersize=6)
        ax.plot(denormalized_truth_x, denormalized_truth_y, '*', color='green', markersize=5)

    plt.title(f"Image ID: {img['id']}, Pedestrian ID: {pedestrian_id}")

    plt.show()


def calculate_euclidean_distance(pred, truth):
    return np.sqrt(np.sum((np.array(pred) - np.array(truth))**2))