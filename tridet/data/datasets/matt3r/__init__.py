# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
from functools import partial

from detectron2.data import DatasetCatalog

from tridet.data.datasets.matt3r.build import register_matt3r_metadata, build_monocular_matt3r_dataset

LOG = logging.getLogger(__name__)

KITTI_ROOT = 'matter'

DATASET_DICTS_BUILDER = {
    # Monocular datasets
    "matt3r_train": (build_monocular_matt3r_dataset, dict(mv3d_split='test')), #TODO change split from test to train
    "matt3r_test": (build_monocular_matt3r_dataset, dict(mv3d_split='test')),
}

METADATA_BUILDER = {name: (register_matt3r_metadata, {}) for name in DATASET_DICTS_BUILDER.keys()}


def register_matt3r_datasets(required_datasets, cfg):
    matt3r_datasets = sorted(list(set(required_datasets).intersection(DATASET_DICTS_BUILDER.keys())))
    if matt3r_datasets:
        LOG.info(f"Matt3r dataset(s): {', '.join(matt3r_datasets)} ")
        for name in matt3r_datasets:
            fn, kwargs = DATASET_DICTS_BUILDER[name]
            kwargs.update({'root_dir': os.path.join(cfg.DATASET_ROOT, KITTI_ROOT)})
            DatasetCatalog.register(name, partial(fn, **kwargs))

            fn, kwargs = METADATA_BUILDER[name]
            kwargs.update({'coco_cache_dir': cfg.TMP_DIR})
            fn(name, **kwargs)
    return matt3r_datasets
