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
# log_model_results
# -----------------------------------------------------------------------------
def log_model_results(performance_data, csv_file="./results/model_performance.csv"):
    """
    Records the performance data into a CSV file, using the experiment timestamp for consistency.

    Args:
        performance_data (dict): Performance data to log.
        csv_file (str): Path to the CSV file for logging performance data.
    """
    results_dir = os.path.dirname(csv_file)
    os.makedirs(results_dir, exist_ok=True)

    # Dataframe Init
    df_new_entry = pd.DataFrame([performance_data])

    # Write performance data to csv
    try:
        if os.path.isfile(csv_file):
            df_existing = pd.read_csv(csv_file, sep=";")
            df_final = pd.concat([df_existing, df_new_entry], ignore_index=True, sort=False)
        else:
            df_final = df_new_entry

        df_final.to_csv(csv_file, index=False, sep=";")
    except Exception as e:
        print(f"Error recording results: {e}")


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





