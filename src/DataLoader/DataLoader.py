import numpy as np
from torch.utils.data import Dataset
import ast


def csv_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        raise Exception("Could not process data, verify csv file")


class KeypointsDataset(Dataset):
    def __init__(self, dataframe):
        self.keypoints = np.array(dataframe['keypoints'].tolist(), dtype=np.float32)
        self.targets = np.array(dataframe['target'].tolist(), dtype=np.float32)
        self.img_ids = dataframe['img_id'].values
        self.bboxes = np.array([ast.literal_eval(bbox) for bbox in dataframe['bbox']], dtype=np.float32)

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return self.img_ids[idx], self.keypoints[idx], self.targets[idx], self.bboxes[idx]
