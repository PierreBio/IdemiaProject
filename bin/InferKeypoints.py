import os
import csv
import numpy as np
import pandas as pd
import torch
from mmpose.apis import init_model
from mmpose.apis import MMPoseInferencer

from src.ImageParser.ImageProcessor import ImageProcessor

def rtm_inference(csv_file, device_name, path_model='checkpoints\\rtmpose_model.onnx', csv_output='data\\train_data_RTMpose.csv'):
    """
    Perform inference on the images in the CSV file using the RTMpose model.
    Args:
        csv_file (str): Path to the CSV file containing the images to perform inference on.
        path_model (str): Path to the RTMpose model.
        device_name (str): Name of the device to use for inference.
        csv_output (str): Path to the output CSV file.
    """
    inferencer = MMPoseInferencer('human')
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
            result_generator = inferencer(img_path, show=False)
            result = next(result_generator)

            predictions = result.get('predictions', [])

            for prediction in predictions[0]:
                keypoints = prediction['keypoints']
                bbox = prediction['bbox']
                keypoints_flat = [kp for sublist in keypoints for kp in sublist]

                if len(keypoints) >= 2:
                    foot1_x, foot1_y = keypoints[-2][0], keypoints[-2][1]
                    foot2_x, foot2_y = keypoints[-1][0], keypoints[-1][1]
                    target = [(foot1_x + foot2_x) / 2, (foot1_y + foot2_y) / 2]
                else:
                    target = [None, None]

                writer.writerow({
                    'img_id': img_id,
                    'pedestrian_id': pedestrian_id,
                    'bbox': str(bbox),
                    'keypoints': str(keypoints_flat),
                    'target': str(target)
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

    rtm_inference('data\\train_data.csv', device_name=device, path_model=onnx_file)

    #https://mmpose.readthedocs.io/en/latest/user_guides/inference.html?highlight=.build#build-a-model