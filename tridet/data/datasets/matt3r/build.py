# Copyright 2021 Toyota Research Institute.  All rights reserved.
import functools
import itertools
import logging
import json
import os
from collections import OrderedDict
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.structures.boxes import BoxMode

from tridet.data import collect_dataset_dicts
from tridet.structures.boxes3d import GenericBoxes3D
from tridet.structures.pose import Pose
from tridet.utils.coco import create_coco_format_cache
from tridet.utils.geometry import project_points3d
from tridet.utils.visualization import float_to_uint8_color

LOG = logging.getLogger(__name__)

VALID_CLASS_NAMES = ("Car", "Pedestrian", "Cyclist", "Van", "Truck")

COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=8)]
COLORMAP = OrderedDict({
    "Car": COLORS[2],  # green
    "Pedestrian": COLORS[1],  # orange
    "Cyclist": COLORS[0],  # blue
    "Van": COLORS[6],  # pink
    "Truck": COLORS[5],  # brown
    "Person_sitting": COLORS[4],  #  purple
    "Tram": COLORS[3],  # red
    "Misc": COLORS[7],  # gray
})

MV3D_SPLIT_MATT3R_REMAP = {
    "train": "training",
    "val": "training",
    "test": "testing",
    "overfit": "training",
    "trainval": "training",
}

class Matt3rDataset(Dataset):
    def __init__(self, root_dir, mv3d_split, class_names, sensors, box2d_from_box3d=False, max_num_items=None):
        self.root_dir = root_dir
        assert mv3d_split in ["train", "val", "trainval", "test"]
        with open(os.path.join(self.root_dir, "mv3d_kitti_splits", "{}.txt".format(mv3d_split))) as _f:
            lines = _f.readlines()
        split = [line.rstrip("\n") for line in lines]
        self._split = split
        if max_num_items is not None:
            self._split = self._split[:min(len(self._split), max_num_items)]
        self._mv3d_split = mv3d_split

        self.class_names = class_names
        self._name_to_id = {name: idx for idx, name in enumerate(class_names)}
        self._sensors = sensors
        if sensors != ('camera_2', ) and not box2d_from_box3d:
            LOG.warning(f"Overwriting 'box2d_from_box3d' from 'False' to 'True' (sensors = {', '.join(sensors)}).")
            box2d_from_box3d = True
        self._box2d_from_box3d = box2d_from_box3d

        self.calibration_table = self._parse_calibration_files()

    def _parse_calibration_files(self):
        """Build calibration table from dataset.
        Calibration details here:
        https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md

        Returns
        -------
        calibration_table: dict, default: None
            Calibration table used for looking up sample-level calibration.
            >>> (K, pose_0S) = self.calibration_table[(calibration_key, datum_name)]

            Calibration key is string filename prefix. (i.e. 007431)
        """
        testing_calib = os.path.join(self.root_dir, f"{MV3D_SPLIT_MATT3R_REMAP[self._mv3d_split]}", "calib", "cameramatrix.txt")
        
        self.K = Matt3rDataset.readNpy(testing_calib)
        print('**********')
        print(self.K)
        print('**********')
       
        per_sample_calibrations = [self._read_calibration_file(index) for index in self._split]
        
        return OrderedDict(per_sample_calibrations)
    
    @staticmethod
    def readNpy(target):
        try:
            with(open(target, 'r')) as f:
                return np.loadtxt(f)
        except:
            return None

    
    def _read_calibration_file(self, index):
        """Reads a calibration file and creates corresponding pose, camera calibration objects.
        Reference frame (world frame) is camera 0.

        Returns
        -------
        calibration_table: dict, default: None
            Calibration table used for looking up sample-level calibration.
            >>> (K, pose_0S) = self.calibration_table[(calibration_key, datum_name)]

            Calibration key is just filename prefix. (i.e. 007431)
        """

        return (index, self.K)

    def __len__(self):
        return len(self._split)

    def __getitem__(self, idx):
        sample_id = self._split[idx]
        sample = OrderedDict()
        for sensor_name in self._sensors:
            sample.update(self._get_sample_data(sample_id, sensor_name))
        return sample

    def _get_sample_data(self, sample_id, sensor_name):
        # Get pose of the sensor (S) from vehicle (V) frame (pose_VS).
        # For KITTI 3D, vehicle frame is equivalent to world frame W, so pose_WS = pose_VS
        intrinsics = self.calibration_table[sample_id]

        kitti_3d_split = MV3D_SPLIT_MATT3R_REMAP[self._mv3d_split]

        datum = {}
        if sensor_name in ("camera_2", "camera_3"):
            datum_dir = "image_2" if sensor_name == "camera_2" else "image_3"
            datum['intrinsics'] = list(intrinsics.flatten())
            # Consistent with COCO format
            datum['file_name'] = os.path.join(self.root_dir, kitti_3d_split, datum_dir, f'{sample_id}.png')
            I = Image.open(datum['file_name'])
            datum['width'], datum['height'] = I.width, I.height
            datum['image_id'] = f'{sample_id}_{sensor_name}'
            datum['sample_id'] = sample_id
        elif sensor_name == "velodyne":
            datum['filename'] = os.path.join(self.root_dir, kitti_3d_split, 'velodyne', f'{sample_id}.bin')
        else:
            raise ValueError(f"Invalid sensor name: {sensor_name}")

        return {sensor_name: datum}
    #TODO read annotations in here
    def get_annotations(self, sample_id, sensor_name):
        try:
            sample_annotations = pd.read_csv(
                os.path.join(
                    self.root_dir, MV3D_SPLIT_MATT3R_REMAP[self._mv3d_split], "label_2", "{}.txt".format(sample_id)
                ),
                delim_whitespace=True,
                header=None
            )
        except pd.errors.EmptyDataError:
            sample_annotations = pd.DataFrame(columns=[i for i in range(16)])

        annotations = []
        for idx, kitti_annotation in sample_annotations.iterrows():
            class_name = kitti_annotation[0]
            if class_name not in self.class_names:
                continue

            annotation = OrderedDict(category_id=self._name_to_id[class_name], instance_id=f'{sample_id}_{idx}')

            annotation.update(self._get_3d_annotation(kitti_annotation, sample_id, sensor_name))
            if self._box2d_from_box3d:
                intrinsics, _ = self.calibration_table[(sample_id, sensor_name)]
                annotation.update(self._compute_box2d_from_box3d(annotation['bbox3d'], intrinsics))
            else:
                assert sensor_name == "camera_2", f"Invalid sensor for 2D annotation: {sensor_name}"
                annotation.update(self._get_2d_annotation(kitti_annotation))

            intrinsics, _ = self.calibration_table[(sample_id, sensor_name)]

            annotations.append(annotation)

        return annotations, sample_annotations

    def _get_3d_annotation(self, label, sample_id, sensor_name):
        """Convert KITTI annotation data frame to 3D bounding box annotations.
        Labels are provided in the reference frame of camera_2.
        NOTE: Annotations are returned in the reference of the requested sensor
        """
        height, width, length = label[8:11]
        x, y, z = label[11:14]
        rotation = label[14]

        # We modify the KITTI annotation axes to our convention  of x-length, y-width, z-height.
        # Additionally, KITTI3D refers to the center of the bottom face of the cuboid, and our convention
        # refers to the center of the 3d cuboid which is offset by height/2. To get a bounding box
        # back in KITTI coordinates, i.e. for evaluation, see `self.convert_to_kitti`.
        box_pose = Pose(
            wxyz=Quaternion(axis=[1, 0, 0], radians=np.pi / 2) * Quaternion(axis=[0, 0, 1], radians=-rotation),
            tvec=np.float64([x, y - height / 2, z])
        )

        if sensor_name != "camera_2":  # "camera_3" or "velodyne"
            _, pose_0S = self.calibration_table[(sample_id, sensor_name)]
            _, pose_02 = self.calibration_table[(sample_id, "camera_2")]
            box_pose = pose_0S * pose_02.inverse() * box_pose

        box3d = GenericBoxes3D(box_pose.quat.elements, box_pose.tvec, [width, length, height])
        vec = box3d.vectorize().tolist()[0]
        distance = float(np.linalg.norm(vec[4:7]))

        return OrderedDict([('bbox3d', vec), ('distance', distance)])

    def _get_2d_annotation(self, label):
        l, t, r, b = label[4:8]
        return OrderedDict(bbox=[l, t, r, b], bbox_mode=BoxMode.XYXY_ABS)

    def _compute_box2d_from_box3d(self, box3d, K):
        box = GenericBoxes3D(box3d[:4], box3d[4:7], box3d[7:])
        corners = project_points3d(box.corners.cpu().numpy()[0], K)

        l, t = corners[:, 0].min(), corners[:, 1].min()
        r, b = corners[:, 0].max(), corners[:, 1].max()
        return OrderedDict(bbox=[l, t, r, b], bbox_mode=BoxMode.XYXY_ABS)


class Matt3rMonocularDataset(Dataset):
    def __init__(self, root_dir, mv3d_split, class_names, sensors, box2d_from_box3d, max_num_items):
        self._matt3r_dset = Matt3rDataset(root_dir, mv3d_split, class_names, sensors, box2d_from_box3d, max_num_items)
        self._sensors = sensors

    def __len__(self):
        return len(self._matt3r_dset) * len(self._sensors)

    def __getitem__(self, idx):
        base_idx, sensor_idx = idx // len(self._sensors), idx % len(self._sensors)
        return self._matt3r_dset[base_idx][self._sensors[sensor_idx]]


@functools.lru_cache(maxsize=1000)
def build_monocular_matt3r_dataset(
    mv3d_split, root_dir, class_names=VALID_CLASS_NAMES, sensors=('camera_2', ), box2d_from_box3d=False, max_num_items=None
):
    dataset = Matt3rMonocularDataset(root_dir, mv3d_split, class_names, sensors, box2d_from_box3d, max_num_items)
    dataset_dicts = collect_dataset_dicts(dataset)
    return dataset_dicts

def register_matt3r_metadata(dataset_name, valid_class_names=VALID_CLASS_NAMES, coco_cache_dir='/tmp/'):
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = valid_class_names
    metadata.thing_colors = [COLORMAP[klass] for klass in metadata.thing_classes]

    metadata.id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.name_to_contiguous_id = {name: idx for idx, name in metadata.contiguous_id_to_name.items()}

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata.json_file = create_coco_format_cache(dataset_dicts, metadata, dataset_name, coco_cache_dir)
    LOG.info(f'COCO json file: {metadata.json_file}')

    metadata.evaluators = ("matt3r_evaluator",)
    metadata.pred_visualizers = ("d2_visualizer", "box3d_visualizer")
    metadata.loader_visualizers = ("d2_visualizer", "box3d_visualizer")
