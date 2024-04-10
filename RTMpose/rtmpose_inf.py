import os
import cv2
import numpy as np
import pandas as pd
import torch
from mmdeploy_runtime import PoseDetector
from src.ImageParser.ImageProcessor import ImageProcessor



def rtm_inference(csv_file, device_name, path_model='RTMpose\\rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip', csv_output='data\\train_data_RTMpose.csv'):
    """
    Perform inference on the images in the CSV file using the RTMpose model.

    Args:
        csv_file (str): Path to the CSV file containing the images to perform inference on.
        path_model (str): Path to the RTMpose model.
        device_name (str): Name of the device to use for inference.
        csv_output (str): Path to the output CSV file.
    """
    detector = PoseDetector(
        model_path=path_model, device_name=device_name, device_id=0)
 
    df = pd.read_csv(csv_file)    
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
            result = detector(img)
        else:
            # converter (x, y, w, h) -> (left, top, right, bottom)
            
            bbox = eval(bbox)
            bbox = np.array(bbox, dtype=int)
            original_bbox = bbox.copy()
            bbox[2:] += bbox[:2]
            result = detector(img, bbox)

        # Display the keypoints on the image
        # _, point_num, _ = result.shape
        # points = result[:, :, :2].reshape(point_num, 2)
        # for [x, y] in points.astype(int):
        #     cv2.circle(img, (x, y), 1, (0, 255, 0), 2)

        # cv2.imwrite(f'{test}_output_pose.png', img)
        
        # Transform result into a list of keypoints
        result = result.reshape(-1).tolist()
        result, bbox = ImageProcessor.normalize_keypoints(result, original_bbox)

        # change  every 3 keypoints value to 2
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
    rtm_inference('data\\train_data.csv',device_name=device)
