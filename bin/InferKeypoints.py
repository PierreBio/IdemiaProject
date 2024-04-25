import os
import csv
import numpy as np
import pandas as pd
import torch
from mmpose.apis import init_model
from mmpose.apis import MMPoseInferencer

from src.ImageParser.ImageProcessor import ImageProcessor
from src.Common.Occlude import Occlude

def format_list(lst):
    """ Formate une liste en chaîne sans espaces après les virgules et sans parenthèses. """
    return "[" + ",".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in lst) + "]"

def occlude_keypoints(keypoints, scores, score_threshold=0.3):
    occluded_keypoints = []
    for (x, y, visibility), score in zip(keypoints, scores):
        if score < score_threshold:
            occluded_keypoints.append([0, 0, 0])  # Marquez le keypoint comme occlus
        else:
            occluded_keypoints.append([x, y, visibility])  # Conservez le keypoint
    return occluded_keypoints

def normalize_and_classify_keypoints(keypoints, scores, bbox):
    x_min, y_min, width, height = bbox[0]
    normalized_keypoints = []

    for kp, score in zip(keypoints, scores):
        x, y = kp
        x_norm = (x - x_min) / width
        y_norm = (y - y_min) / height
        visibility = 2 if score > 0.5 else 1 if score > 0.2 else 0
        normalized_keypoints.append([x_norm, y_norm, visibility])

    return normalized_keypoints

def rtm_inference(csv_file,
                  device_name,
                  path_model="./checkpoints/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth",
                  config_path="./configs/config_rtm_pose.py",
                  csv_output='data\\train_data_RTMpose.csv'):
    """
    Perform inference on the images in the CSV file using the RTMpose model.
    Args:
        csv_file (str): Path to the CSV file containing the images to perform inference on.
        device_name (str): Name of the device to use for inference.
        path_model (str): Path to the RTMpose model.
        config_path (str): Path to the RTMpose config.
        csv_output (str): Path to the output CSV file.
    """
    inferencer = MMPoseInferencer(
        pose2d=config_path,
        pose2d_weights=path_model
    )
    df = pd.read_csv(csv_file)

    with open(csv_output, 'w', newline='') as output_file:
        fieldnames = ['img_id', 'pedestrian_id', 'bbox', 'keypoints', 'target']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in df.iterrows():
            img_id = row['img_id']
            pedestrian_id = row['pedestrian_id']
            img_filename = f"{img_id}_{pedestrian_id}.jpg"
            img_path = os.path.join(os.getcwd(), "data", "images", "train", img_filename)

            if not os.path.exists(img_path):
                print(f"Image file not found: {img_path}")
                continue

            print(f"Performing inference on image {img_id} with pedestrian ID {pedestrian_id}...")
            result_generator = inferencer(img_path, show=True)
            result = next(result_generator)

            predictions = result.get('predictions', [])

            if predictions:
                prediction = predictions[0][0]
                bbox = prediction['bbox']
                keypoints = prediction['keypoints']
                keypoints = keypoints[:-2]
                scores = prediction['keypoint_scores']
                normalized_keypoints = normalize_and_classify_keypoints(keypoints, scores, bbox)
                occluded_keypoints = occlude_keypoints(normalized_keypoints, scores, score_threshold=0.5)

                keypoints_flat = [item for sublist in normalized_keypoints for item in sublist]

                if len(normalized_keypoints) >= 2:
                    foot1_x, foot1_y, _ = normalized_keypoints[-2]
                    foot2_x, foot2_y, _ = normalized_keypoints[-1]
                    target = [(foot1_x + foot2_x) / 2, (foot1_y + foot2_y) / 2]
                else:
                    target = [None, None]

                writer.writerow({
                    'img_id': img_id,
                    'pedestrian_id': pedestrian_id,
                    'bbox': str(bbox),
                    'keypoints': str(keypoints_flat),
                    'target': str(target) if target != [None, None] else "[]"
                })

    print(f"Results saved to {csv_output}")

if __name__ == '__main__':
    # Set device name to 'cuda' if you have a GPU
    device = 'cpu'
    # if torch.cuda.is_available():
    #     device = 'cuda'
    config_path = "./configs/config_rtm_pose.py"
    checkpoint_path = "./checkpoints/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"

    model = init_model(config_path, checkpoint_path, device=device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    input_tensor = torch.randn(1, 3, 256, 192)
    onnx_file = "./checkpoints/rtmpose_model.onnx"

    torch.onnx.export(model, input_tensor, onnx_file, opset_version=11,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    rtm_inference('data\\train_data.csv', device_name=device, path_model=checkpoint_path, config_path=config_path)

    #https://mmpose.readthedocs.io/en/latest/user_guides/inference.html?highlight=.build#build-a-model