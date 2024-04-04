# import os
# import mmengine
from mmengine.config import Config
from mmengine.runner import Runner
from typing import Dict
from mmengine.fileio import dump
import numpy as np
from mmpose.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
# import torch
from typing import Dict, Optional


def custom_results2json(self, keypoints: Dict[int, list],
                        outfile_prefix: str) -> str:
    """Dump the keypoint detection results to a COCO style json file.

    Args:
        keypoints (Dict[int, list]): Keypoint detection results
            of the dataset.
        outfile_prefix (str): The filename prefix of the json files. If the
            prefix is "somepath/xxx", the json files will be named
            "somepath/xxx.keypoints.json",

    Returns:
        str: The json file name of keypoint results.
    """
    # the results with category_id
    cat_results = []

    for _, img_kpts in keypoints.items():
        _keypoints = np.array(
            [img_kpt['keypoints'] for img_kpt in img_kpts])
        num_keypoints = self.dataset_meta['num_keypoints']
        # collect all the person keypoints in current image
        _keypoints = _keypoints.reshape(-1, num_keypoints * 3)

        result = []
        for img_kpt, keypoint in zip(img_kpts, _keypoints):
            res = {
                'image_id': img_kpt['img_id'],
                'category_id': img_kpt['category_id'],
                'keypoints': keypoint.tolist(),
                'score': float(img_kpt['score']),
                'id': img_kpt['id']
            }
            if 'bbox' in img_kpt:
                res['bbox'] = img_kpt['bbox'].tolist()
            result.append(res)

        cat_results.extend(result)

    res_file = f'{outfile_prefix}.keypoints.json'
    dump(cat_results, res_file, sort_keys=True, indent=4)


@TRANSFORMS.register_module()
class Occlude(BaseTransform):
    """Occlude a patch

    Args:
        patch_size (float):
    """

    def __init__(self, patch_size: float = 5) -> None:
        super().__init__()

        self.patch_size = patch_size

    def transform(self, results: Dict) -> Optional[dict]:
        """The transform function of :class:`GetBBoxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        results['img'][:self.patch_size, :self.patch_size, :] = np.zeros(
            (self.patch_size, self.patch_size, 3))
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(patch_size={self.patch_size})'
        return repr_str


# load config
cfg = Config.fromfile("configs/conf1.py")

# build the runner from config
runner = Runner.from_cfg(cfg)

# Overwrite results2json to save box id in output file
runner.test_evaluator.metrics[0].results2json = custom_results2json.__get__(
    runner.test_evaluator.metrics[0])

runner.test()
