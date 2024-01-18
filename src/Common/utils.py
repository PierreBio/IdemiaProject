import csv
import pandas as pd
from tabulate import tabulate
import math
import random
import numpy as np


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
    Applies occlusion to a batch of keypoints based on the specified parameters.

    Args:
        inputs (array-like): Array of keypoints for each data point in the batch.
        weight_position (str): "lower_body", "upper_body", or "" for random occlusion.
        weight_value (float): Weight value to determine occlusion probability.
        min_visible_threshold (int): Minimum visible keypoints in an image.

    Returns:
        np.array: Array of keypoints with occlusion applied.
    """
    occluded_inputs = []

    # Define ranges for upper and lower body keypoints
    upper_body_range = range(0, 33)  # Upper body contains 11 keypoints
    lower_body_range = range(33, 45)  # Lower body contains 4 keypoints (excluding ankles)

    for keypoints in inputs:
        # Count non-visible keypoints
        non_visible_count = keypoints.count(0) // 3

        # Skip this keypoints list if non-visible keypoints exceed the threshold
        if non_visible_count > min_visible_threshold:
            occluded_inputs.append(keypoints)
            continue

        # Initialize occluded keypoints
        occluded_keypoints = keypoints.copy()

        for i in range(len(occluded_keypoints) // 3):
            if non_visible_count > min_visible_threshold:
                break

            # Determine occlusion probability
            occlusion_chance = weight_value if ((weight_position == "lower_body" and i in lower_body_range) or
                                                (weight_position == "upper_body" and i in upper_body_range)) else 1 - weight_value

            # Apply occlusion
            if random.random() < occlusion_chance:
                occluded_keypoints[3 * i:3 * i + 3] = [0, 0, 0]
                non_visible_count += 1

        occluded_inputs.append(occluded_keypoints)

    return np.array(occluded_inputs)


def apply_box_occlusion(img_ids, inputs, targets=None, occlusion_chance=0.8, range_occlusion=(0.5, 1)):
    """
    Applies occlusion to keypoints based on the corresponding image annotations.

    Args:
        img_ids (array-like): Array of image IDs corresponding to each data point in the batch.
        inputs (array-like): Array of keypoints for each data point in the batch.
        targets (array-like, optional): Array of target values for each data point in the batch.
        occlusion_chance (float): Probability of applying occlusion to a given set of keypoints.
        range_occlusion (tuple): Range (min, max) for the scaling factor of occlusion.

    Returns:
        tuple: A tuple containing occluded inputs, and optionally occluded targets if targets are not None.
               Format: (occluded_inputs, occluded_targets) or occluded_inputs if targets is None.

    Note:
        This function assumes access to a COCO-style database (`self.__coco_db`) to fetch annotations
        based on image IDs, and a method `self.__normalize_keypoints` for normalizing keypoints.
    """
    return apply_keypoints_occlusion(inputs, "upper_body")
    # occluded_inputs = []
    # occluded_targets = []
    #
    # for idx, img_id in enumerate(img_ids):
    #     keypoints = inputs[idx]
    #     target = targets[idx] if targets is not None else None
    #
    #     img_ann = self.__coco_db.loadAnns(img_id)
    #     img_ann = img_ann[0]
    #     box = img_ann["bbox"]
    #     box_occluded = box.copy()
    #
    #     random_value = random.uniform(range_occlusion[0], range_occlusion[1])
    #     if random.random() < occlusion_chance:
    #         box_occluded[3] = box_occluded[3] * random_value
    #
    #     normalized_kps = self.__normalize_keypoints(keypoints, box_occluded)
    #
    #     occluded_inputs.append(normalized_kps)
    #     if target is not None:
    #         occluded_targets.append(target)
    #
    # return (np.array(occluded_inputs), np.array(occluded_targets)) if targets is not None else np.array(
    #     occluded_inputs)