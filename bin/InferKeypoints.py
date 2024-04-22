import os
import cv2
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

    result_generator = inferencer("./data/images/train/431_435945.jpg", show=True)
    result = next(result_generator)
    print(result)
    print("GOOOOOOO")
    df = pd.read_csv(csv_file)
    print(df.shape)
    for index,row in df.iterrows():
        img_id = row['img_id']
        img_filename = f"{str(img_id).zfill(12)}.jpg"
        img_path = os.path.join(os.getcwd(), "data", "images", "train", img_filename)

        if not os.path.exists(img_path):
            df.drop(index, inplace=True)
            continue

        img = cv2.imread(img_path)
        bbox = row['bbox']

        print(f"Performing inference on image {img_id}...")
        #on effectue l'inférence
        if bbox is None:
            result_generator = inferencer(img)
            result = next(result_generator)
        else:
            bbox = eval(bbox)
            bbox = np.array(bbox, dtype=int)
            original_bbox = bbox.copy()
            bbox[2:] += bbox[:2]
            result_generator = inferencer(img, bbox)
            result = next(result_generator)
            print(result)


        result = result.reshape(-1).tolist()
        result, bbox = ImageProcessor.normalize_keypoints(result, original_bbox)

        for i in range(len(result)):
            if (i + 1) % 3 == 0:
                result[i] = 2

        # Replace the keypoints in the dataframe
        df.at[index, 'keypoints'] = result
    # Save the dataframe to a new CSV file
    df.to_csv(csv_output, index=False)

if __name__ == '__main__':
    # Set device name to 'cuda' if you have a GPU
    device = 'cpu'
    # if torch.cuda.is_available():
    #     device = 'cuda'
    config_path = "./configs/config_rtm_pose.py"
    checkpoint_path = "./checkpoints/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"

    print("HELLLO")
    model = init_model(config_path, checkpoint_path, device=device)
    print("BYYEEE1")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("BYYEEE2")

    input_tensor = torch.randn(1, 3, 256, 192)
    onnx_file = "./checkpoints/rtmpose_model.onnx"
    print("BYYYEEE3")
    torch.onnx.export(model, input_tensor, onnx_file, opset_version=11,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    rtm_inference('data\\train_data.csv', device_name=device, path_model=onnx_file)

    #https://mmpose.readthedocs.io/en/latest/user_guides/inference.html?highlight=.build#build-a-model