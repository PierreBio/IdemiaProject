import os
import torch
import ast
import cv2
import pandas as pd
import numpy as np
import yaml
from mmpose.apis import MMPoseInferencer

from bin.VisualiseUtils import visualize_specific_pedestrian_local, calculate_euclidean_distance
from src.Models.Mlp import MLP
from src.DataLoader.DataLoader import csv_string_to_list
from src.ImageParser.ImageProcessor import ImageProcessor
from src.Common.Occlude import Occlude

def load_calibration_data(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    K = data['K']
    R = data['R']
    T = data['T']
    return K, R, T

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def calculate_depth(K, R, T, output):
    u, v = output[0], output[1]
    H_real = 1.75
    f_x = K[0][0]
    f_y = K[1][1]
    c_x = K[0][2]
    c_y = K[1][2]
    Z = H_real * f_y / (v - c_y)

    return Z

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

def classify_keypoints(keypoints, scores, bbox):
    x_min, y_min, width, height = bbox[0][:4]
    normalized_keypoints = []

    keypoints = keypoints[:15]  # Adjust this number to match the model's expected input
    scores = scores[:15]

    for kp, score in zip(keypoints, scores):
        x, y = kp
        x_norm = (x - x_min) / width
        y_norm = (y - y_min) / height
        visibility = 2 if score > 0.5 else 1 if score > 0.2 else 0
        normalized_keypoints.append([x_norm, y_norm, visibility])

    return normalized_keypoints

def load_model():
    input_size = 45
    output_size = 2
    layers = [256, 128, 64, 32]

    model = MLP(input_size, output_size, layers)
    model.load_state_dict(torch.load("models/20240306_214252_LR0.0001_BS16/best_model_epoch_1_rmse_0.1210.pth"))
    model.eval()
    return model

def process_video(video_path, model, K, R, T):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = preprocess_frame(frame)

        config_path = "./configs/config_rtm_pose.py"
        path_model = "./checkpoints/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"

        inferencer = MMPoseInferencer(
            pose2d=config_path,
            pose2d_weights=path_model
        )

        result_generator = inferencer(input_frame, show=False)
        result = next(result_generator)

        predictions = result.get('predictions', [])

        if predictions:
            prediction = predictions[0][0]
            bbox = prediction['bbox']
            keypoints = prediction['keypoints']
            scores = prediction['keypoint_scores']

            print(f"BBox: {bbox}")  # Debug print for BBox
            normalized_keypoints = classify_keypoints(keypoints, scores, bbox)
            keypoints_flat = [item for sublist in normalized_keypoints for item in sublist]
            input_tensor = torch.tensor(keypoints_flat, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred = model(input_tensor).squeeze().tolist()
                print(pred)
                height, width = frame.shape[:2]
                x, y = int(pred[0] * width), int(pred[1] * height)

        Z = calculate_depth(K, R, T, [x, y])
        print(Z)
        print(f"Predicted coordinates: X={x}, Y={y}, Z={Z}")
        new_width = int(width / 4)
        new_height = int(height / 4)

        # Draw the BBox
        x_min, y_min, width, height = bbox[0]  # Utilisez directement bbox
        x_min, y_min = int(x_min), int(y_min)
        width, height = int(width), int(height)
        cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2)

        for kp in normalized_keypoints:
            x_kp, y_kp = int(kp[0] * width) + x_min, int(kp[1] * height) + y_min
            cv2.circle(frame, (x_kp, y_kp), 3, (255, 0, 0), -1)

        if len(keypoints) >= 2:
            left_foot = keypoints[-2]  # Utilisez les deux derniers keypoints pour les pieds
            right_foot = keypoints[-1]
            midpoint_x = int((left_foot[0] + right_foot[0]) / 2)
            midpoint_y = int((left_foot[1] + right_foot[1]) / 2)
            cv2.circle(frame, (midpoint_x, midpoint_y), 5, (0, 255, 0), -1)

        resized_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow('Frame', resized_frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def opencv_matrix_constructor(loader, node):
    """Construct a numpy array from the YAML node."""
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    if 'rows' in mapping and 'cols' in mapping:
        mat = mat.reshape(mapping['rows'], mapping['cols'])
    return mat

def convert_bbox(bbox_str):
    cleaned_str = bbox_str.replace("([", "").replace("],)", "")
    if cleaned_str.endswith(','):
        cleaned_str = cleaned_str[:-1]
    return f"[{cleaned_str}]"

def main():
    #df = pd.read_csv('./data/train_data_RTMpose.csv')
    #df['bbox'] = df['bbox'].apply(convert_bbox)
    #df.to_csv('./data/train_data_RTMpose_formatted.csv', index=False)

    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)
    calibration_file = 'config/calibration_chessboard.yml'
    video_path = 'data/videos/out_1_blurred.mp4'

    K, rvec, T = load_calibration_data(calibration_file)
    R, _ = cv2.Rodrigues(rvec)
    model = load_model()
    process_video(video_path, model, K, R, T)

if __name__ == '__main__':
    main()