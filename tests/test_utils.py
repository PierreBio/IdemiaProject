import pytest
from src.Common.utils import visualize_csv_stats
from io import StringIO
import math


# Helper function to mimic a CSV file
def mock_csv(data: str):
    return StringIO(data)


# Test with a valid CSV data
@pytest.mark.parametrize("data, expected", [
    (
        """img_id,pedestrian_id,keypoints,target\n
        1,1,"[1, 1, 1, 1]","[42, 42, 0, 42, 42, 0, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2]","[42, 42]"\n
        2,2,"[1, 1, 1, 1]","[43, 43, 0, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2]","[43, 43]"\n""",
        {
            "Total images": 2,
            'Average keypoints occluded per image': 2,
            'Max visible keypoints': 14,
            'Min visible keypoints': 13,
            'Percentage of images with occlusion': 100.0,
            "Most frequently occluded keypoint": "nose"
        }
    ),
])
def test_occluded_data(data, expected):
    file_path = mock_csv(data)
    stats = visualize_csv_stats(file_path)
    assert stats == expected


@pytest.mark.parametrize("data, expected", [
    (
        """img_id,pedestrian_id,keypoints,target\n
        1,1,"[1, 1, 1, 1]","[42, 42, 2, 42, 42, 2, 42, 42, 0, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2, 42, 42, 2]","[42, 42]"\n
        2,2,"[1, 1, 1, 1]","[43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2, 43, 43, 2]","[43, 43]"\n""",
        {
            "Total images": 2,
            'Average keypoints occluded per image': 1,
            'Max visible keypoints': 15,
            'Min visible keypoints': 14,
            'Percentage of images with occlusion': 50.0,
            "Most frequently occluded keypoint": "right_eye"
        }
    ),
])
def test_non_occluded_data(data, expected):
    file_path = mock_csv(data)
    stats = visualize_csv_stats(file_path)
    assert stats == expected
