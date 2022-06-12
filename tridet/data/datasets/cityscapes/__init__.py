# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os
from functools import partial

from detectron2.data import DatasetCatalog

from tridet.data.datasets.cityscapes.build import register_cityscapes_metadata, build_monocular_cityscapes_dataset

LOG = logging.getLogger(__name__)

ROOT = 'cityscapes'

DATASET_DICTS_BUILDER = {
    # Monocular datasets
    "cityscapes_train": (build_monocular_cityscapes_dataset, dict(mv3d_split='train')), #TODO change split from test to train
    "cityscapes_val": (build_monocular_cityscapes_dataset, dict(mv3d_split='val')),
    "cityscapes_test": (build_monocular_cityscapes_dataset, dict(mv3d_split='test')),
}

METADATA_BUILDER = {name: (register_cityscapes_metadata, {}) for name in DATASET_DICTS_BUILDER.keys()}

def register_cityscapes_dataset(required_datasets, cfg):
    cityscapes_datasets = sorted(list(set(required_datasets).intersection(DATASET_DICTS_BUILDER.keys())))
    if cityscapes_datasets:
        LOG.info(f"CityScapes dataset(s): {', '.join(cityscapes_datasets)} ")
        for name in cityscapes_datasets:
            fn, kwargs = DATASET_DICTS_BUILDER[name]
            kwargs.update({'root_dir': os.path.join(cfg.DATASET_ROOT, ROOT)})
            DatasetCatalog.register(name, partial(fn, **kwargs))

            fn, kwargs = METADATA_BUILDER[name]
            kwargs.update({'coco_cache_dir': cfg.TMP_DIR})
            fn(name, **kwargs)
    return cityscapes_datasets
