from collections import OrderedDict
import random
# Copyright 2021 Toyota Research Institute.  All rights reserved.
import itertools
import json
import math
import os
import warnings
from collections import OrderedDict
from functools import partial

from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import BoxMode
from pyquaternion import Quaternion
import numpy as np
from tridet.structures.pose import Pose
import pandas as pd
from iopath.common.file_io import PathManager


BBOX3D_PREDICTION_FILE = "bbox3d_predictions.json"

def convert_3d_box_to_kitti(box):
    """Convert a single 3D bounding box (GenericBoxes3D) to KITTI convention. i.e. for evaluation. We
    assume the box is in the reference frame of camera_2 (annotations are given in this frame).

    Usage:
        >>> box_camera_2 = pose_02.inverse() * pose_0V * box_velodyne
        >>> kitti_bbox_params = convert_3d_box_to_kitti(box_camera_2)

    Parameters
    ----------
    box: GenericBoxes3D
        Box in camera frame (X-right, Y-down, Z-forward)

    Returns
    -------
    W, L, H, x, y, z, rot_y, alpha: float
        KITTI format bounding box parameters.
    """
    assert len(box) == 1

    quat = Quaternion(*box.quat.cpu().tolist()[0])
    tvec = box.tvec.cpu().numpy()[0]
    sizes = box.size.cpu().numpy()[0]

    # Re-encode into KITTI box convention
    # Translate y up by half of dimension
    tvec += np.array([0., sizes[2] / 2.0, 0])

    inversion = Quaternion(axis=[1, 0, 0], radians=np.pi / 2).inverse
    quat = inversion * quat

    # Construct final pose in KITTI frame (use negative of angle if about positive z)
    if quat.axis[2] > 0:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=-quat.angle), tvec=tvec)
        rot_y = -quat.angle
    else:
        kitti_pose = Pose(wxyz=Quaternion(axis=[0, 1, 0], radians=quat.angle), tvec=tvec)
        rot_y = quat.angle

    # Construct unit vector pointing in z direction (i.e. [0, 0, 1] direction)
    # The transform this unit vector by pose of car, and drop y component, thus keeping heading direction in BEV (x-z grid)
    v_ = np.float64([[0, 0, 1], [0, 0, 0]])
    v_ = (kitti_pose * v_)[:, ::2]

    # Getting positive theta angle (we define theta as the positive angle between
    # a ray from the origin through the base of the transformed unit vector and the z-axis
    theta = np.arctan2(abs(v_[1, 0]), abs(v_[1, 1]))

    # Depending on whether the base of the transformed unit vector is in the first or
    # second quadrant we add or subtract `theta` from `rot_y` to get alpha, respectively
    alpha = rot_y + theta if v_[1, 0] < 0 else rot_y - theta
    # Bound from [-pi, pi]
    if alpha > np.pi:
        alpha -= 2.0 * np.pi
    elif alpha < -np.pi:
        alpha += 2.0 * np.pi
    alpha = np.around(alpha, decimals=2)  # KITTI precision

    # W, L, H, x, y, z, rot-y, alpha
    return sizes[0], sizes[1], sizes[2], tvec[0], tvec[1], tvec[2], rot_y, alpha


class MatterEvaluator:
    def __init__(self, output_dir=None, dataset_name=None):
        self._output_dir = output_dir
        self.dataset_name = dataset_name

        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        class_names = metadata.thing_classes
        id_to_name = metadata.contiguous_id_to_name

        self._dataset_dicts = {dikt['file_name']: dikt for dikt in dataset_dicts}
        self._id_to_name = id_to_name
        self._class_names = class_names

    def reset(self):
        # List[Dict], each key'ed by category (str) + vectorized 3D box (10) + 2D box (4) + score (1) + file name (str)
        self._predictions_as_json = []

        self._predictions_kitti_format = []
        self._groundtruth_kitti_format = []
    
    def evaluate(self):
        predictions_as_json = self._predictions_as_json
        predictions_kitti_format = self._predictions_kitti_format
        groundtruth_kitti_format = self._groundtruth_kitti_format

        # Write prediction file as JSON.
        PathManager().mkdirs(self._output_dir)
        file_path = os.path.join(self._output_dir, BBOX3D_PREDICTION_FILE)
        with open(file_path, 'w') as f:
            json.dump(predictions_as_json, f, indent=4)
        results = OrderedDict()
        print(self.outputs)
        # results[random.randint(0,5)] = self.inputs
        # results[random.randint(5,10)] = self.outputs
        return results

    def process(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        for input_per_image, pred_per_image in zip(inputs, outputs):
            pred_classes = pred_per_image['instances'].pred_classes
            pred_boxes = pred_per_image['instances'].pred_boxes.tensor
            pred_boxes3d = pred_per_image['instances'].pred_boxes3d
            # pred_boxes3d = pred_per_image['instances'].pred_box3d_as_vec
            scores = pred_per_image['instances'].scores
            scores_3d = pred_per_image['instances'].scores_3d

            file_name = input_per_image['file_name']
            image_id = input_per_image['image_id']

            # predictions
            predictions_kitti = []
            # for class_id, box3d_as_vec, score, box2d in zip(pred_classes, pred_boxes3d, scores, pred_boxes):
            for class_id, box3d, score_3d, box2d, score in zip(
                pred_classes, pred_boxes3d, scores_3d, pred_boxes, scores
            ):
                # class_name = self._metadata.thing_classes[class_id]
                class_name = self._class_names[class_id]

                box3d_as_vec = box3d.vectorize()[0].cpu().numpy()

                pred = OrderedDict(
                    category_id=int(class_id),  # COCO instances
                    category=class_name,
                    bbox3d=box3d_as_vec.tolist(),
                    # COCO instances uses "XYWH". Aligning with it as much as possible
                    bbox=BoxMode.convert(box2d.tolist(), from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYWH_ABS),
                    score=float(score),
                    score_3d=float(score_3d),
                    file_name=file_name,
                    image_id=image_id  # COCO instances
                )
                self._predictions_as_json.append(pred)

                # prediction in KITTI format.
                W, L, H, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box3d)
                l, t, r, b = box2d.tolist()
                predictions_kitti.append([
                    class_name, -1, -1, alpha, l, t, r, b, H, W, L, x, y, z, rot_y,
                    float(score_3d)
                ])
            self._predictions_kitti_format.append(pd.DataFrame(predictions_kitti))

            # groundtruths
            gt_dataset_dict = self._dataset_dicts[file_name]

            if "annotations" not in gt_dataset_dict:
                # test set
                continue

            raw_kitti_annotations = gt_dataset_dict.get('raw_kitti_annotations', None)
            if raw_kitti_annotations is not None:
                self._groundtruth_kitti_format.append(raw_kitti_annotations)
            else:
                # Otherwise, use the same format as predictions (minus 'score').
                groundtruth_kitti = []
                for anno in gt_dataset_dict['annotations']:
                    # class_name = self._metadata.thing_classes[anno['category_id']]
                    class_name = self._class_names[anno['category_id']]

                    # groundtruth in KITTI format.
                    box2d = BoxMode.convert(anno['bbox'], from_mode=anno['bbox_mode'], to_mode=BoxMode.XYXY_ABS)
                    box3d = GenericBoxes3D.from_vectors([anno['bbox3d']])
                    W, L, H, x, y, z, rot_y, alpha = convert_3d_box_to_kitti(box3d)
                    l, t, r, b = box2d
                    groundtruth_kitti.append([class_name, -1, -1, alpha, l, t, r, b, H, W, L, x, y, z, rot_y])
                self._groundtruth_kitti_format.append(pd.DataFrame(groundtruth_kitti))