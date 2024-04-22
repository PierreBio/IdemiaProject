import os
import subprocess
import requests

import csv
import mmengine
import numpy as np
from typing import Dict
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.fileio import dump
from zipfile import ZipFile

from src.Common.Occlude import Occlude

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
        cat_results = []

        for _, img_kpts in keypoints.items():
            _keypoints = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            num_keypoints = self.dataset_meta['num_keypoints']
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

def download_file(url, path_local):
    os.makedirs(os.path.dirname(path_local), exist_ok=True)
    response = requests.get(url)
    with open(path_local, 'wb') as f:
        f.write(response.content)

def download_and_extract(url, path_local):
        nom_fichier_zip = url.split('/')[-1]
        chemin_complet_zip = os.path.join(path_local, nom_fichier_zip)

        if not os.path.exists(chemin_complet_zip):
            os.makedirs(path_local, exist_ok=True)
            print(f"Téléchargement de {url}...")
            response = requests.get(url)
            with open(chemin_complet_zip, 'wb') as f:
                f.write(response.content)
            print(f"{nom_fichier_zip} téléchargé.")
        else:
            print(f"Le fichier {chemin_complet_zip} existe déjà.")

        chemin_destination = os.path.join(path_local, 'coco')

        if not os.path.exists(os.path.join(chemin_destination, 'val2017')):
            print(f"Extraction de {nom_fichier_zip}...")
            with ZipFile(chemin_complet_zip, 'r') as zip_ref:
                zip_ref.extractall(chemin_destination)
            print("Extraction terminée.")
        else:
            print("Les données ont déjà été extraites.")

def download_rtm_pose_config():
    config_url = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-t_8xb256-420e_coco-256x192.py"
    checkpoint_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"
    images_url = "http://images.cocodataset.org/zips/val2017.zip"

    config_path = "./configs/rtmpose-t_8xb256-420e_coco-256x192.py"
    checkpoint_path = "./checkpoints/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"
    extraction_path = "data"

    if not os.path.exists(config_path):
        print(f"Téléchargement de {config_url}...")
        download_file(config_url, config_path)
    else:
        print(f"Le fichier {config_path} existe déjà.")

    if not os.path.exists(checkpoint_path):
        print(f"Téléchargement de {checkpoint_url}...")
        download_file(checkpoint_url, checkpoint_path)
    else:
        print(f"Le fichier {checkpoint_path} existe déjà.")

    download_and_extract(images_url, extraction_path)

def main():
    download_rtm_pose_config()

    cfg = Config.fromfile("./configs/config_rtm_pose.py")
    runner = Runner.from_cfg(cfg)
    runner.test_evaluator.metrics[0].results2json = custom_results2json.__get__(runner.test_evaluator.metrics[0])
    runner.test()

if __name__ == "__main__":
    main()