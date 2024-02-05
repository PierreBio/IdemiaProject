import logging
import os
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import numpy as np
import torch


# -----------------------------------------------------------------------------
# ImageProcessor
# -----------------------------------------------------------------------------
class ImageProcessor:
    """ Initializes an Image Processor Object
                : path to the Coco db annotation file default is empty
                         str

    Attributes:
        __db_file_path : str
            Path to the database annotation file.
        __coco_db : str
            Coco DB object.
        __parsed_data : str
            List of parsed data in the form [img_id, annotation_id, keypoints, target]
            for each data points.
    """

    def __init__(self, db_ann_file: str = ""):
        self.__db_file_path = db_ann_file
        self.__coco_db = COCO(db_ann_file) if db_ann_file else None

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
                x = (x - x0) / width
                y = (y - y0) / height
        return norm_keypoints

    def parse_images_with_dynamic_occlusion(self, image_ids, threshold, occlusion_chance=0.8):
        """
        Parses images, applying dynamic occlusion to keypoints based on a given threshold and occlusion chance.

        Args:
            image_ids (list): List of selected image IDs to process.
            threshold (float): Required percentage of visible keypoints.
            occlusion_chance (float): Chance of applying occlusion to the keypoints.
        """
        parsed_data = []

        for img_id in image_ids:
            img_anns = self.__coco_db.loadAnns(self.__coco_db.getAnnIds(img_id))

            for img_ann in img_anns:
                try:
                    # Apply dynamic occlusion to keypoints
                    occluded_keypoints, occluded_bbox = self.apply_dynamic_occlusion(img_ann["keypoints"],
                                                                                     img_ann["bbox"], occlusion_chance)

                    # Check if the occluded keypoints meet the visibility threshold
                    if self.__kps_visibility_check(occluded_keypoints, threshold):
                        # Assuming a method to compute the target based on occluded keypoints if necessary
                        target = self.compute_target(occluded_keypoints)

                        # Append the processed data to the parsed_data list
                        parsed_data.append([img_id, img_ann["id"], occluded_bbox, occluded_keypoints, target])
                except Exception as e:
                    logging.error(f"Error while processing image {img_id} | pedestrian {img_ann['id']}: {e}")

        self.__parsed_data = parsed_data

    # -----------------------------------------------------------------------------
    # __parse_images
    # -----------------------------------------------------------------------------
    def __parse_images(self, image_ids, threshold) -> None:
        """ Parsing Images following project criteria.

        Args:
            image_ids: Selected Images
            threshold: Required % of visible keypoints

        Returns:
            None
        """
        # Iterating through images
        for img_id in image_ids:
            img_anns = self.__coco_db.loadAnns(
                self.__coco_db.getAnnIds(img_id))

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
                            self.__parsed_data.append(
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

    # -----------------------------------------------------------------------------
    # apply_dynamic_occlusion
    # -----------------------------------------------------------------------------
    def apply_dynamic_occlusion(self, keypoints, bbox, occlusion_chance=0.8):
        # Occlusion decision
        occlusion_type = random.choices(["none", "box", "keypoints"],
                                        weights=[1-occlusion_chance, occlusion_chance/2, occlusion_chance/2], k=1)[0]

        if occlusion_type == "box":
            return ImageProcessor.normalize_keypoints(keypoints, bbox), bbox
        elif occlusion_type == "keypoints":
            return self.apply_keypoints_occlusion(keypoints, "", 0.7, 5)
        else:
            # No occlusion applied, just normalize the keypoints
            return self.normalize_keypoints(keypoints, bbox), bbox

    # -----------------------------------------------------------------------------
    # parse_annotation_file
    # -----------------------------------------------------------------------------
    def parse_annotation_file(self, ann_file_path="", cat_names=None, threshold=70) -> list:
        """ Parse the given annotation file.

        Args:
            ann_file_path (str): path to the Coco db annotation file default is empty, default ""
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
        # Using Class annotation file if none was given
        if not ann_file_path:
            ann_file_path = self.__db_file_path

        # Sanity check
        if not os.path.isfile(ann_file_path):
            raise FileExistsError(
                f"File {ann_file_path} does not exist, exiting..")

        # Initializing Coco DB
        self.__coco_db = COCO(ann_file_path)

        # Cat Name check
        if cat_names is None:
            raise Exception(f"Unsupported category names: {cat_names}")

        cat_ids = self.__coco_db.getCatIds(catNms=cat_names)
        img_ids = self.__coco_db.getImgIds(catIds=cat_ids)

        # Parsing selected images
        self.__parse_images(img_ids, threshold)

        return self.__parsed_data

    # -----------------------------------------------------------------------------
    # apply_keypoints_occlusion
    # -----------------------------------------------------------------------------
    @staticmethod
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
        occluded_inputs = inputs.clone()

        # Define ranges for upper and lower body keypoints
        upper_body_range = range(0, 11)
        lower_body_range = range(11, 15)

        for idx, keypoints in enumerate(occluded_inputs):
            keypoints_reshaped = keypoints.view(-1, 3)  # Reshape to have 3 elements per row
            non_visible_count = torch.sum(torch.all(keypoints_reshaped == 0, dim=1)).item()

            # Skip this keypoints if non-visible keypoints exceed the threshold
            if non_visible_count > min_visible_threshold:
                occluded_inputs.append(keypoints)
                continue

            for i in range(keypoints_reshaped.size(0)):
                if non_visible_count > min_visible_threshold:
                    break  # Stop if we reach the visibility threshold

                if (weight_position == "lower_body" and i in lower_body_range or
                        weight_position == "upper_body" and i in upper_body_range):
                    occlusion_chance = weight_value
                else:
                    occlusion_chance = 1 - weight_value

                if random.random() < occlusion_chance:
                    occluded_inputs[idx, 3 * i:3 * i + 3] = torch.tensor([0, 0, 0], dtype=torch.float32)
                    non_visible_count += 1

        return occluded_inputs

    @staticmethod
    def apply_box_occlusion(inputs, boxes, targets, occlusion_chance=0.8, range_occlusion=(0.5, 1)):
        """
        Applies occlusion to keypoints based on the corresponding image annotations.

        Args:
            inputs (list): List of keypoints for each data point in the batch.
            boxes (list): List of bounding boxes for each data point in the batch.
            targets (list): List of target values for each data point in the batch that may need adjustment.
            occlusion_chance (float): Probability of applying occlusion to a given set of keypoints.
            range_occlusion (tuple): Range (min, max) for the scaling factor of occlusion.

        Returns:
            tuple: A tuple containing occluded inputs, and optionally occluded targets if targets are not None.
                   Format: (occluded_inputs, occluded_targets) or occluded_inputs if targets is None.

        Note:
            This function assumes access to a COCO-style database (`self.__coco_db`) to fetch annotations
            based on image IDs, and a method `self.normalize_keypoints` for normalizing keypoints.
        """
        occluded_inputs = []
        occluded_targets = []

        for keypoints, box, target in zip(inputs, boxes, targets):
            # Cloning box to avoid mutation
            box_occluded = list(box)

            if random.random() < occlusion_chance:
                # Randomly scale a dimension of the box for occlusion
                scale_factor = random.uniform(*range_occlusion)
                box_occluded[3] *= scale_factor

            # Normalize keypoints with the occluded box
            normalized_kps = ImageProcessor.normalize_keypoints(keypoints, box_occluded)
            occluded_inputs.append(normalized_kps)

            # Normalize target with the occluded box
            normalized_target = ImageProcessor.normalize_keypoints(target, box_occluded)
            occluded_targets.append(normalized_target)

        return occluded_inputs, occluded_targets
