# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import cv2
import numpy as np
from mmdeploy_runtime import PoseDetector


def infer_keypoints(path_images, path_model='RTMpose\\rtmpose-cpu.zip', device_name='cpu'):

    detector = PoseDetector(
        model_path=path_model, device_name=device_name, device_id=0)

    # if args.bbox is None:
    #     result = detector(img)
    # else:
    #     # converter (x, y, w, h) -> (left, top, right, bottom)
    #     print(args.bbox)
    #     bbox = np.array(args.bbox, dtype=int)
    #     bbox[2:] += bbox[:2]
    #     result = detector(img, bbox)

    # On parcourt chaque image contenu dans path_img
    # On récupére chaque fichier contenu dans le dossier path_images

    for file in os.listdir(path_images):
        # Vérifier si le fichier est une image
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = os.path.join(path_images, file)
            img = cv2.imread(image)
            result = detector(img)
            # save the result in a csv
            np.savetxt('data\\test.csv', result, delimiter=',')

            # Display the keypoints on the image
            # _, point_num, _ = result.shape
            # points = result[:, :, :2].reshape(point_num, 2)
            # for [x, y] in points.astype(int):
            #     cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

            # cv2.imwrite('output_pose.png', img)


if __name__ == '__main__':
    infer_keypoints('data\\images\\train\\')
