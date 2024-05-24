import torch
import cv2
import numpy as np
import yaml
from mmpose.apis import MMPoseInferencer

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
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]
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

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
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
            keypoints = keypoints[:-2]
            scores = prediction['keypoint_scores']
            normalized_keypoints = normalize_and_classify_keypoints(keypoints, scores, bbox)

            with torch.no_grad():
                output = model(normalized_keypoints)
                print(output)
                x, y = int(output[0]), int(output[1])

        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        Z = calculate_depth(K, R, T, output)
        print(f"Predicted coordinates: X={x}, Y={y}, Z={Z}")

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
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

def main():
    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)
    calibration_file = 'config/calibration_chessboard.yml'
    model_path = 'models/20240306_214252_LR0.0001_BS16/best_model_epoch_1_rmse_0.1210.pth'
    video_path = 'data/out_0_blurred.mp4'

    K, rvec, T = load_calibration_data(calibration_file)
    R, _ = cv2.Rodrigues(rvec)
    model = load_model(model_path)
    process_video(video_path, model, K, R, T)

if __name__ == '__main__':
    main()