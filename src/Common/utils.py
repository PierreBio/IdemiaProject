import csv
import pandas as pd
from tabulate import tabulate
import math
import random
import numpy as np
import torch
import os
from datetime import datetime


# -----------------------------------------------------------------------------
# save_to_csv
# -----------------------------------------------------------------------------
def save_to_csv(file, headers, data_list):
    """ Saving given headers & data to given file.

    Args:
        file: path to the output file
        headers: CSV file headers
        data_list: Data to be saved

    Returns:
        None
    """
    # Define file  headers

    with open(file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Writing headers & data
        writer.writerow(headers)
        for row in data_list:
            writer.writerow(row if isinstance(row, list) else [row])


# -----------------------------------------------------------------------------
# record_results
# -----------------------------------------------------------------------------
def record_results(performance_data, csv_file="./results/model_performance.csv"):
    """
    Records the performance of hyperparameters into a CSV file.

    Args:
        performance_data (dict): A dictionary containing the performance data.
        csv_file (str, optional): Path to the CSV file where performance data will be saved.
                                  Defaults to "./results/model_performance.csv".

    The function creates a new file or appends to an existing one, organizing the data by ascending RMSE values.
    """

    # Ensure the results directory exists
    results_dir = os.path.dirname(csv_file)
    os.makedirs(results_dir, exist_ok=True)

    # Add the current date to the data
    performance_data["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert the data to a DataFrame
    df = pd.DataFrame([performance_data])

    # Append to existing file or write a new file
    if os.path.isfile(csv_file):
        existing_df = pd.read_csv(csv_file, sep=";")
        df = pd.concat([existing_df, df])

    # Sort by RMSE and save
    df.sort_values(by="RMSE", inplace=True)
    df.to_csv(csv_file, index=False, header=True, sep=";")


# -----------------------------------------------------------------------------
# visualize_csv_stats
# -----------------------------------------------------------------------------
def visualize_csv_stats(file_path):
    """ Visualizes statistics of a keypoints dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        stats: dictionary containing the csv detailed statistics
    """
    keypoint_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                      "left_knee", "right_knee"]

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Convert keypoints from string to list
    df['keypoints'] = df['keypoints'].apply(eval)

    # Calculate visibility for each keypoint
    df['visible_keypoints'] = df['keypoints'].apply(lambda kps: [kps[i*3+2] for i in range(len(keypoint_names))])
    df['occluded_keypoints'] = df['visible_keypoints'].apply(lambda vis: [1 if v == 0 else 0 for v in vis])

    # Calculate the most frequently occluded keypoint
    occlusion_counts = [sum(kps[i] for kps in df['occluded_keypoints']) for i in range(len(keypoint_names))]
    most_occluded_index = occlusion_counts.index(max(occlusion_counts))
    most_frequent_occluded_kp = keypoint_names[most_occluded_index] if occlusion_counts[most_occluded_index] > 0 else None

    stats = {
        'Total images': len(df),
        'Average keypoints occluded per image': math.ceil(sum(occlusion_counts) / len(df)),
        'Max visible keypoints': df['visible_keypoints'].apply(lambda kps: sum(v > 0 for v in kps)).max(),
        'Min visible keypoints': df['visible_keypoints'].apply(lambda kps: sum(v > 0 for v in kps)).min(),
        'Percentage of images with occlusion': round(100 * sum(any(v == 0 for v in vis) for vis in df['visible_keypoints']) / len(df), 2),
        'Most frequently occluded keypoint': most_frequent_occluded_kp,
    }

    # Print table & return results
    print(tabulate(stats.items(), headers=["Statistic", "Value"], floatfmt=".2f", stralign="left"))
    return stats


def apply_keypoints_occlusion(inputs,
                              weight_position="",
                              weight_value=0.7,
                              min_visible_threshold=5):
    """
    Applies occlusion to a batch of keypoints based on specified parameters.

    Args:
        inputs (Tensor): A batch of keypoints, where each keypoint has 3 values.
        weight_position (str): "lower_body", "upper_body", or "" for random occlusion.
        weight_value (float): Weight value to determine occlusion probability.
        min_visible_threshold (int): Minimum number of visible keypoints in an image.

    Returns:
        Tensor: Batch of keypoints with occlusion applied.
    """
    occluded_inputs = []

    # Define ranges for upper and lower body keypoints
    upper_body_range = range(0, 11)
    lower_body_range = range(11, 15)

    for keypoints in inputs:
        keypoints_reshaped = keypoints.view(-1, 3)  # Reshape to have 3 elements per row
        non_visible_count = torch.sum(torch.all(keypoints_reshaped == 0, dim=1)).item()

        # Skip this keypoints if non-visible keypoints exceed the threshold
        if non_visible_count > min_visible_threshold:
            occluded_inputs.append(keypoints)
            continue

        occluded_keypoints = keypoints.clone()

        for i in range(keypoints_reshaped.size(0)):
            if non_visible_count > min_visible_threshold:
                break

            occlusion_chance = weight_value if ((weight_position == "lower_body" and i in lower_body_range) or
                                                (weight_position == "upper_body" and i in upper_body_range)) else 1 - weight_value

            if random.random() < occlusion_chance:
                occluded_keypoints[3*i:3*i+3] = torch.tensor([0, 0, 0])
                non_visible_count += 1

        occluded_inputs.append(occluded_keypoints)

    return torch.stack(occluded_inputs)


