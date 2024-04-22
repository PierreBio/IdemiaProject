import numpy as np

from mmpose.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import torch
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
        results['img'][:self.patch_size, :self.patch_size, :] = np.zeros((self.patch_size, self.patch_size, 3))
        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f'(patch_size={self.patch_size})'
        return repr_str
