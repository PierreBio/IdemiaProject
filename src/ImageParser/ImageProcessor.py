import logging
import os
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
import torch
import pandas as pd


# -----------------------------------------------------------------------------
# ImageProcessor
# -----------------------------------------------------------------------------
class ImageProcessor:
    """ Initializes an Image Processor Object
                : path to the Coco db annotation file default is empty
                         str

    """

    def __init__(self):
        pass

    # -----------------------------------------------------------------------------
    # __kps_visibility_check
    # -----------------------------------------------------------------------------
    @staticmethod
    def __kps_visibility_check(keypoint_list, threshold) -> bool:
        """ Checks keypoints visibility against given threshold.

        Args:
            keypoint_list: Coco keypoint list
            threshold: Required % of visible keypoints

        Returns:
            True if threshold is met, False otherwise
        """
        # Extracting visibility values from kps
        v_values = keypoint_list[2::3]

        # Counting number of non-zero visibility values
        visible_count = sum(v != 0 for v in v_values)

        # Checking visibility count against threshold
        total_kps = len(v_values)
        visible_percentage = (visible_count / total_kps) * 100

        return visible_percentage >= threshold

    # -----------------------------------------------------------------------------
    # normalize_keypoints
    # -----------------------------------------------------------------------------
    @staticmethod
    def normalize_keypoints(keypoints, bbox):
        """ Normalizes keypoints to be relative to the bounding box.

        Args:
            keypoints: List of keypoints for an image.
            bbox: Bounding box with [x0, y0, width, height].

        Returns:
            List of normalized keypoints.
        """
        x0, y0, width, height = bbox
        if width == 0 or height == 0:
            raise Exception(f"Error while normalizing: width={width} | height={height}")

        norm_keypoints = []
        for i in range(0, len(keypoints), 3):
            x, y, v = keypoints[i:i+3]
            if v != 0:
                x_norm = (x - x0) / width
                y_norm = (y - y0) / height
                norm_keypoints.extend([x_norm, y_norm, v])
            else:
                norm_keypoints.extend([x, y, v])
        return norm_keypoints

    @staticmethod
    def apply_dynamic_occlusion_to_csv(parsed_data, **kwargs):
        modified_data = []
        for data_point in parsed_data:
            img_id, annotation_id, bbox, keypoints, target = data_point
            occlusion_type = random.choice(['no_occlusion', 'box', 'keypoints'])

            if occlusion_type == 'box':
                occlusion_chance = kwargs.get('bb_occlusion_chance', 0.7)
                box_scale_factor = kwargs.get('bb_scale_factor', (0.5, 1))
                occluded_bbox, occluded_keypoints, occluded_target = ImageProcessor.apply_box_occlusion(keypoints,
                                                                                                        bbox,
                                                                                                        target,
                                                                                                        occlusion_chance,
                                                                                                        box_scale_factor)
                modified_data.append([img_id, annotation_id, occluded_bbox, occluded_keypoints, occluded_target])
            elif occlusion_type == 'keypoints':
                weight_position = kwargs.get('kp_weight_position', "upper_body")
                weight_value = kwargs.get('kp_weight_value', 0.7)
                min_visible_threshold = kwargs.get('kp_min_threshold', 5)
                occluded_keypoints = ImageProcessor.apply_keypoints_occlusion(keypoints,
                                                                              weight_position,
                                                                              weight_value,
                                                                              min_visible_threshold)
                modified_data.append([img_id, annotation_id, bbox, occluded_keypoints, target])
            elif occlusion_type == "no_occlusion":
                # do nothing
                pass

        return modified_data

    # -----------------------------------------------------------------------------
    # __parse_images
    # -----------------------------------------------------------------------------
    def __parse_images(self, coco_db, image_ids, threshold):
        """ Parsing Images following project criteria.

        Args:
            image_ids: Selected Images
            threshold: Required % of visible keypoints

        Returns:
            List of data.
        """
        parsed_data = []

        # Iterating through images
        for img_id in image_ids:
            img_anns = coco_db.loadAnns(
                coco_db.getAnnIds(img_id))

            # Iterating through image annotations
            for img_ann in img_anns:
                # Normalize keypoints
                try:
                    normalized_kps = self.normalize_keypoints(img_ann["keypoints"], img_ann["bbox"])

                    # Checking feets visibility (for scientific purpose only)
                    # Feets are last 2 sets of kps
                    if normalized_kps[-1] != 0 and normalized_kps[-4] != 0:
                        # Computing % of visible kps
                        if self.__kps_visibility_check(normalized_kps[:-6], threshold):
                            logging.debug(f"[ImageParse]: ACCEPTED Img {img_id}")

                            # Computing distance between 2 feets
                            target = [(normalized_kps[-3] + normalized_kps[-6]) / 2,
                                      (normalized_kps[-5] + normalized_kps[-2]) / 2]

                            # Adding accepted data to parsed_data
                            parsed_data.append(
                                [img_id, img_ann["id"], img_ann["bbox"], normalized_kps[:-6], target])
                        else:
                            logging.debug(
                                f"[ImageParse]: REJECTED, Not enough kps visible for image {img_id}")
                    else:
                        logging.debug(
                            f"[ImageParse]: REJECTED, Feets not visible for image {img_id}")

                except Exception as e:
                    print(f"Error while processing image {img_id} | pedestrian {img_ann['id']}: {e}")
                    pass

        return parsed_data

    # -----------------------------------------------------------------------------
    # parse_annotation_file
    # -----------------------------------------------------------------------------
    def parse_annotation_file(self, ann_file_path, cat_names=None, threshold=70) -> list:
        """ Parse the given annotation file.

        Args:
            ann_file_path (str): path to the Coco db annotation file default is empty
            cat_names (list of str): Image Categories (i.e, person), default None
            threshold (int): Required % of visible keypoints, default 70

        Returns:
            list of accepted data:
                each data points are in the form [img_id, annotation_id, keypoints, target]
                where keypoints are a list of keypoints without the feets keypoints
                where target is the distance between 2 feets
        Raises:
            FileExistsError: if file path is erroneous

        Note:
            Calling this method, resets all previously parsed data. Internal parsed data
            will be set to the output of this method
        """
        # Sanity check
        if not os.path.isfile(ann_file_path):
            raise FileExistsError(
                f"File {ann_file_path} does not exist, exiting..")

        # Initializing Coco DB
        coco_db = COCO(ann_file_path)

        # Cat Name check
        if cat_names is None:
            raise Exception(f"Unsupported category names: {cat_names}")

        cat_ids = coco_db.getCatIds(catNms=cat_names)
        img_ids = coco_db.getImgIds(catIds=cat_ids)

        # Parsing selected images
        parsed_data = self.__parse_images(coco_db, img_ids, threshold)

        return parsed_data

    @staticmethod
    def apply_keypoints_occlusion(keypoints, weight_position="", weight_value=0.7,
                                  min_visible_threshold=5):
        """
        Applies occlusion to keypoints and normalizes both keypoints and target based on specified parameters.

        Args:
            keypoints (list): List of keypoints for an image, where each keypoint has 3 values (x, y, visibility).
            weight_position (str): "lower_body", "upper_body", or "" for random occlusion.
            weight_value (float): Weight value to determine occlusion probability.
            min_visible_threshold (int): Minimum number of visible keypoints in an image.

        Returns:
            tuple: Tuple containing occluded and normalized keypoints, and normalized target.
        """
        visible_count = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)

        # If already below the threshold, return original keypoints and target
        if visible_count <= min_visible_threshold:
            return keypoints

        # Define ranges for upper and lower body keypoints
        upper_body_range = range(0, 11)
        lower_body_range = range(11, 15)

        occluded_keypoints = keypoints.copy()
        for i in range(0, len(keypoints), 3):
            if visible_count <= min_visible_threshold:
                break  # Stop if we reach the visibility threshold

            v = keypoints[i+2]
            should_occlude = False

            if v > 0:
                # Determine if the keypoint should be occluded based on weight_position and weight_value
                if weight_position == "lower_body" and (i // 3) in lower_body_range or \
                        weight_position == "upper_body" and (i // 3) in upper_body_range:
                    should_occlude = random.random() < weight_value
                else:
                    should_occlude = random.random() < (1 - weight_value)

            if should_occlude:
                visible_count -= 1
                occluded_keypoints[i] = 0
                occluded_keypoints[i+1] = 0
                occluded_keypoints[i+2] = 0

        return occluded_keypoints

    @staticmethod
    def apply_box_occlusion(keypoints, bbox, target, occlusion_chance=0.8, range_occlusion=(0.5, 1)):
        """
        Applies occlusion to keypoints and the target by modifying the bounding box.

        Args:
            keypoints (list): List of keypoints for an image.
            bbox (list): Bounding box for an image [x, y, width, height].
            target (list): Target keypoint to be normalized alongside the keypoints.
            occlusion_chance (float): Probability of applying occlusion to the bounding box.
            range_occlusion (tuple): Range (min, max) for the scaling factor of occlusion.

        Returns:
            tuple: Tuple containing occluded and normalized keypoints, occluded bounding box, and normalized target.
        """
        occluded_box = list(bbox)  # Make a copy of the bbox to avoid modifying the original
        if random.random() < occlusion_chance:
            # Randomly scale the height of the box for occlusion
            scale_factor = random.uniform(*range_occlusion)
            occluded_box[3] *= scale_factor  # Apply occlusion by modifying the height

        # Normalize keypoints and target with the occluded box
        normalized_keypoints = ImageProcessor.normalize_keypoints(keypoints, occluded_box)

        # TODO: Adding visibility data to the target to be able to normalize it
        #       and then removing it (this is a temp solution)
        new_target = target + [2]
        normalized_target = ImageProcessor.normalize_keypoints(new_target, occluded_box)

        return occluded_box, normalized_keypoints, normalized_target[:-1]

    # # -----------------------------------------------------------------------------
    # # apply_keypoints_occlusion
    # # -----------------------------------------------------------------------------
    # @staticmethod
    # def apply_keypoints_occlusion(inputs,
    #                               weight_position="",
    #                               weight_value=0.7,
    #                               min_visible_threshold=5):
    #     """
    #     Applies occlusion to a batch of keypoints based on specified parameters.
    #
    #     Args:
    #         inputs (Tensor): A batch of keypoints, where each keypoint has 3 values.
    #         weight_position (str): "lower_body", "upper_body", or "" for random occlusion.
    #         weight_value (float): Weight value to determine occlusion probability.
    #         min_visible_threshold (int): Minimum number of visible keypoints in an image.
    #
    #     Returns:
    #         Tensor: Batch of keypoints with occlusion applied.
    #     """
    #     occluded_inputs = inputs.clone()
    #
    #     # Define ranges for upper and lower body keypoints
    #     upper_body_range = range(0, 11)
    #     lower_body_range = range(11, 15)
    #
    #     for idx, keypoints in enumerate(occluded_inputs):
    #         keypoints_reshaped = keypoints.view(-1, 3)  # Reshape to have 3 elements per row
    #         non_visible_count = torch.sum(torch.all(keypoints_reshaped == 0, dim=1)).item()
    #
    #         # Skip this keypoints if non-visible keypoints exceed the threshold
    #         if non_visible_count > min_visible_threshold:
    #             occluded_inputs.append(keypoints)
    #             continue
    #
    #         for i in range(keypoints_reshaped.size(0)):
    #             if non_visible_count > min_visible_threshold:
    #                 break  # Stop if we reach the visibility threshold
    #
    #             if (weight_position == "lower_body" and i in lower_body_range or
    #                     weight_position == "upper_body" and i in upper_body_range):
    #                 occlusion_chance = weight_value
    #             else:
    #                 occlusion_chance = 1 - weight_value
    #
    #             if random.random() < occlusion_chance:
    #                 occluded_inputs[idx, 3 * i:3 * i + 3] = torch.tensor([0, 0, 0], dtype=torch.float32)
    #                 non_visible_count += 1
    #
    #     return occluded_inputs
    #
    # @staticmethod
    # def apply_box_occlusion(inputs, boxes, targets, occlusion_chance=0.8, range_occlusion=(0.5, 1)):
    #     """
    #     Applies occlusion to keypoints based on the corresponding image annotations.
    #
    #     Args:
    #         inputs (list): List of keypoints for each data point in the batch.
    #         boxes (list): List of bounding boxes for each data point in the batch.
    #         targets (list): List of target values for each data point in the batch that may need adjustment.
    #         occlusion_chance (float): Probability of applying occlusion to a given set of keypoints.
    #         range_occlusion (tuple): Range (min, max) for the scaling factor of occlusion.
    #
    #     Returns:
    #         tuple: A tuple containing occluded inputs, and optionally occluded targets if targets are not None.
    #                Format: (occluded_inputs, occluded_targets) or occluded_inputs if targets is None.
    #
    #     Note:
    #         This function assumes access to a COCO-style database (`self.__coco_db`) to fetch annotations
    #         based on image IDs, and a method `self.normalize_keypoints` for normalizing keypoints.
    #     """
    #     occluded_inputs = []
    #     occluded_targets = []
    #
    #     for keypoints, box, target in zip(inputs, boxes, targets):
    #         # Cloning box to avoid mutation
    #         box_occluded = list(box)
    #
    #         if random.random() < occlusion_chance:
    #             # Randomly scale a dimension of the box for occlusion
    #             scale_factor = random.uniform(*range_occlusion)
    #             box_occluded[3] *= scale_factor
    #
    #         # Normalize keypoints with the occluded box
    #         normalized_kps = ImageProcessor.normalize_keypoints(keypoints, box_occluded)
    #         occluded_inputs.append(normalized_kps)
    #
    #         # Normalize target with the occluded box
    #         normalized_target = ImageProcessor.normalize_keypoints(target, box_occluded)
    #         occluded_targets.append(normalized_target)
    #
    #     return occluded_inputs, occluded_targets
