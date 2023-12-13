import logging
import os
import random
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io


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
        self.__coco_db = None
        self.__parsed_data = []

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
                img_kps = img_ann["keypoints"]

                # Checking feets visibility (for scientific purpose only)
                # Feets are last 2 sets of kps
                if img_kps[-1] != 0 and img_kps[-4] != 0:
                    # Computing % of visible kps
                    if self.__kps_visibility_check(img_kps[:-6], threshold):
                        logging.debug(f"[ImageParse]: ACCEPTED Img {img_id}")

                        # Computing distance between 2 feets
                        target = [(img_kps[-3] + img_kps[-6]) / 2,
                                  (img_kps[-5] + img_kps[-2]) / 2]

                        # Adding accepted data to parsed_data
                        self.__parsed_data.append(
                            [img_id, img_ann["id"], img_kps[:-6], target])
                    else:
                        logging.debug(
                            f"[ImageParse]: REJECTED, Not enough kps visible for image {img_id}")
                else:
                    logging.debug(
                        f"[ImageParse]: REJECTED, Feets not visible for image {img_id}")

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

        # Resetting parsed data
        self.__parsed_data = []

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
    # generate_occluded_data
    # -----------------------------------------------------------------------------
    def generate_occluded_data(self, weight_position="", weight_value=0.7, include_original_data=True):
        """ Augments the data by duplicating each entry with occluded keypoints.

        Args:
            weight_position (str): "lower_body" or "upper_body" to bias the occlusion, "" for completely random
            weight_value (float): weight value with default = 0.7,
                                  non targeted parts will have the weight of 1-weight_value
            include_original_data (bool): If True (default), include original data alongside occluded data

        Returns:
            list: Augmented data with each original entry followed by an occluded version

        Note:
            Calling this method, does not affect the internal parsed data. The returned augmented
            data will be lost if not stored.
        """
        # Return var Init
        if include_original_data:
            augmented_data = self.__parsed_data.copy()
        else:
            augmented_data = []

        # Upper body contains 11 kps, lower body contains 4 kps (excluding ankles)
        upper_body_range = range(0, 33)
        lower_body_range = range(33, 45)

        for data_point in self.__parsed_data:
            img_id, ann_id, keypoints, target = data_point

            # Count non-visible keypoints
            non_visible_count = keypoints.count(0) // 3

            # Skip this keypoints list if non-visible keypoints exceed the threshold
            if non_visible_count > 5:
                print("skipping keypoint list")
                continue

            # Initializing occluded kps
            occluded_keypoints = keypoints.copy()

            for i in range(len(occluded_keypoints) // 3):
                # Determine occlusion proba based on weight
                if ((weight_position == "lower_body" and i in lower_body_range) or
                        (weight_position == "upper_body" and i in upper_body_range)):
                    occlusion_chance = weight_value
                else:
                    occlusion_chance = 1 - weight_value

                # Apply occlusion randomly based on calculated chance
                if random.random() < occlusion_chance:
                    occluded_keypoints[3*i:3*i+3] = [0, 0, 0]

            augmented_data.append([img_id, ann_id, occluded_keypoints, target])

        return augmented_data

    # -----------------------------------------------------------------------------
    # display_images
    # -----------------------------------------------------------------------------
    def display_images(self, images_ids, annotations):
        """ Display Images with their annotations.

        Args:
            images_ids: Images to display, referenced by their COCO Id
            annotations: Image annotation in COCO form

        Returns:
            None
        """
        for img_ids in images_ids:
            # Load image
            img = self.__coco_db.loadImgs(img_ids)[0]

            # Display img from url
            # TODO: add parameter to display from DB or URL
            plt.imshow(io.imread(img["coco_url"]))

            # Display kps
            self.__coco_db.showAnns(annotations)

            plt.axis('off')
            plt.show()

    # -----------------------------------------------------------------------------
    # parsed data getter
    # -----------------------------------------------------------------------------
    @property
    def parsed_data(self) -> list:
        return self.__parsed_data
