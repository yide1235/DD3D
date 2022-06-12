# Copyright 2021 Toyota Research Institute.  All rights reserved.
import functools
import itertools
import logging
import json
import os
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
import glob

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

VALID_CLASS_NAMES = ("car", "person", "truck", "bicycle")

COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=8)]
COLORMAP = OrderedDict({
    "car": COLORS[2],  # green
    "person": COLORS[1],  # orange
    "bicycle": COLORS[0],  # blue
    "Van": COLORS[6],  # pink
    "truck": COLORS[5],  # brown
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

class cityScapesDataset(Dataset):
    def __init__(self, root_dir, mv3d_split, class_names, sensors, box2d_from_box3d=False, max_num_items=None):
        self.root_dir = root_dir
        assert mv3d_split in ["train", "val", "trainval", "test"]
        image_directory = os.path.join(root_dir, 'leftImg8bit', mv3d_split)
        images = glob.glob(f"{image_directory}/**/*.png")
        self._split = []
        self._annotation_files = {}
        for sample_id, img_path in enumerate(images):
            self._split.append((sample_id, img_path))
            label_filename = img_path.replace('leftImg8bit','gtBbox3d')
            label_filename = label_filename.replace('.png', '.json')
            self._annotation_files[sample_id] = label_filename
        
        # self._split = self._split[:20]
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
        with Pool(min(cpu_count(), 4)) as _proc:
            per_sample_calibrations = itertools.chain.from_iterable(
                [_proc.map(self._read_calibration_file, self._annotation_files)]
            )
        return OrderedDict(per_sample_calibrations)
    
    @staticmethod
    def readNpy(target):
        try:
            with(open(target, 'r')) as f:
                return np.loadtxt(f)
        except:
            return None

    
    def _read_calibration_file(self, sample_id):
        """Reads a calibration file and creates corresponding pose, camera calibration objects.
        Reference frame (world frame) is camera 0.

        Returns
        -------
        calibration_table: dict, default: None
            Calibration table used for looking up sample-level calibration.
            >>> (K, pose_0S) = self.calibration_table[(calibration_key, datum_name)]

            Calibration key is just filename prefix. (i.e. 007431)
        """
        label_path = self._annotation_files[sample_id]
        try:
            with open(label_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(e)
            data = []

        sensor_data = data['sensor']
        fx = sensor_data['fx']
        fy = sensor_data['fy']
        u0 = sensor_data['u0']
        v0 = sensor_data['v0']
        k = np.eye(3)
        k[0,0] = fx
        k[1,1] = fy
        k[0,2] = u0
        k[1,2] = v0
        img_width = data['imgWidth']
        img_height = data['imgHeight']
        return (sample_id, (k, (img_width, img_height)))

    def __len__(self):
        return len(self._split)

    def __getitem__(self, idx):
        sample_id, img_path = self._split[idx]
        sample = OrderedDict()
        for sensor_name in self._sensors:
            sample.update(self._get_sample_data(sample_id, sensor_name, img_path))
        return sample

    def _get_sample_data(self, sample_id, sensor_name, img_path):
        (intrinsics, (img_width, img_height)) = self.calibration_table[sample_id]
        datum = {}
        datum['intrinsics'] = list(intrinsics.flatten())
        # Consistent with COCO format
        datum['file_name'] = img_path
        datum['width'], datum['height'] = img_width, img_height
        datum['image_id'] = f'{sample_id}' #_{sensor_name}
        datum['sample_id'] = sample_id

        # We define extrinsics as the pose of sensor (S) from the Velodyne (V)
        # _, pose_0V = self.calibration_table[(sample_id, 'velodyne')]
        # _, pose_0S = self.calibration_table[(sample_id, sensor_name)]
        # extrinsics = pose_0V.inverse() * pose_0S
        # datum['extrinsics'] = {'wxyz': extrinsics.quat.elements.tolist(), 'tvec': extrinsics.tvec.tolist()}
        # datum['extrinsics'] = np.eye(3)
        annotations, raw_cityscapes_annotations = self.get_annotations(sample_id, sensor_name)
        datum.update({'annotations': annotations})

        # if sensor_name == "camera_2":
        datum.update({"raw_cityscapes_annotations": raw_cityscapes_annotations})

        return {'camera_2': datum} # not sure if sensor name is used somewhere else. keeping it for now.
    
    def get_annotations(self, sample_id, sensor_name):
        try:
            path = self._annotation_files[sample_id]
            with open(path, 'r') as f:
                sample_annotations = json.load(f)
        except:
            sample_annotations = []

        annotations = []
        for idx, cityScapes_annotation in enumerate(sample_annotations['objects']):
            class_name = cityScapes_annotation['label']
            if class_name not in self.class_names:
                continue

            annotation = OrderedDict(category_id=self._name_to_id[class_name], instance_id=f'{sample_id}_{idx}')

            annotation.update(self._get_3d_annotation(cityScapes_annotation, sample_id, sensor_name))
            if self._box2d_from_box3d:
                intrinsics, _ = self.calibration_table[(sample_id, sensor_name)]
                annotation.update(self._compute_box2d_from_box3d(annotation['bbox3d'], intrinsics))
            else:
                assert sensor_name == "camera_2", f"Invalid sensor for 2D annotation: {sensor_name}"
                annotation.update(self._get_2d_annotation(cityScapes_annotation))

            intrinsics, _ = self.calibration_table[sample_id]
            annotations.append(annotation)

        return annotations, sample_annotations

    def _get_3d_annotation(self, label, sample_id, sensor_name):
        """Convert KITTI annotation data frame to 3D bounding box annotations.
        Labels are provided in the reference frame of camera_2.
        NOTE: Annotations are returned in the reference of the requested sensor
        """
        _3dlabel = label['3d']
        length, width, height = _3dlabel['dimensions']
        x, y, z = _3dlabel['center']
        rotation = _3dlabel['rotation']
        rotation = np.array(rotation)
        rotation = rotation / np.linalg.norm(rotation)
        
        box_pose = Pose(
            wxyz=Quaternion(rotation),
            tvec=np.float64([x, y, z])
        )

        box3d = GenericBoxes3D(box_pose.quat.elements, box_pose.tvec, [width, length, height])
        vec = box3d.vectorize().tolist()[0]
        distance = float(np.linalg.norm(vec[4:7]))

        return OrderedDict([('bbox3d', vec), ('distance', distance)])


    def _get_2d_annotation(self, label):
        _2d_label = label['2d']['amodal']
        xmin, ymin, w, h = _2d_label
        return OrderedDict(bbox=[xmin, ymin, w, h], bbox_mode=BoxMode.XYWH_ABS)

    def _compute_box2d_from_box3d(self, box3d, K):
        box = GenericBoxes3D(box3d[:4], box3d[4:7], box3d[7:])
        corners = project_points3d(box.corners.cpu().numpy()[0], K)

        l, t = corners[:, 0].min(), corners[:, 1].min()
        r, b = corners[:, 0].max(), corners[:, 1].max()
        return OrderedDict(bbox=[l, t, r, b], bbox_mode=BoxMode.XYXY_ABS)


class CityScapesMonocularDataset(Dataset):
    def __init__(self, root_dir, mv3d_split, class_names, sensors, box2d_from_box3d, max_num_items):
        self._matt3r_dset = cityScapesDataset(root_dir, mv3d_split, class_names, sensors, box2d_from_box3d, max_num_items)
        self._sensors = sensors

    def __len__(self):
        return len(self._matt3r_dset) * len(self._sensors)

    def __getitem__(self, idx):
        base_idx, sensor_idx = idx // len(self._sensors), idx % len(self._sensors)
        return self._matt3r_dset[base_idx][self._sensors[sensor_idx]]


@functools.lru_cache(maxsize=1000)
def build_monocular_cityscapes_dataset(
    mv3d_split, root_dir, class_names=VALID_CLASS_NAMES, sensors=('camera_2', ), box2d_from_box3d=False, max_num_items=None
):
    dataset = CityScapesMonocularDataset(root_dir, mv3d_split, class_names, sensors, box2d_from_box3d, max_num_items)
    dataset_dicts = collect_dataset_dicts(dataset)
    return dataset_dicts

def register_cityscapes_metadata(dataset_name, valid_class_names=VALID_CLASS_NAMES, coco_cache_dir='/tmp/'):
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = valid_class_names
    metadata.thing_colors = [COLORMAP[klass] for klass in metadata.thing_classes]

    metadata.id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.contiguous_id_to_name = {idx: klass for idx, klass in enumerate(metadata.thing_classes)}
    metadata.name_to_contiguous_id = {name: idx for idx, name in metadata.contiguous_id_to_name.items()}

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata.json_file = create_coco_format_cache(dataset_dicts, metadata, dataset_name, coco_cache_dir)
    LOG.info(f'COCO json file: {metadata.json_file}')

    metadata.evaluators = ("cityscapes_evaluator",)
    metadata.pred_visualizers = ("d2_visualizer", "box3d_visualizer")
    metadata.loader_visualizers = ("d2_visualizer", "box3d_visualizer")
