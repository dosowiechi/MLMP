
import sys
import os

from mmengine.runner import Runner
from mmcv.transforms.processing import Resize
from mmcv.transforms.loading import LoadImageFromFile

from mmseg.datasets import CityscapesDataset
from mmseg.datasets.transforms.loading import LoadAnnotations
from mmseg.datasets.transforms.formatting import PackSegInputs

from mmengine.registry import TRANSFORMS
from mmengine.registry import DATASETS


# Register the modules with TRANSFORMS
TRANSFORMS.register_module(module=LoadImageFromFile, force=True)
TRANSFORMS.register_module(module=Resize, force=True)
TRANSFORMS.register_module(module=LoadAnnotations, force=True)
TRANSFORMS.register_module(module=PackSegInputs, force=True)
DATASETS.register_module(module=CityscapesDataset, force=True)

batch_size = 2
num_workers = 4
persistent_workers = True
sampler = {'type': 'DefaultSampler', 
           'shuffle': False} ### #TODO: should this be True? ==> i think david set it to False to avoid shuffling the data for adaptation
dataset =  {'type': 'CityscapesDataset', 
            'data_root': '/export/livia/home/vision/Mnoori/old_home/data/segmentation/cityscapes', 
            'data_prefix': {'img_path': 'leftImg8bit/val', 'seg_map_path': 'gtFine/val'}, 
            'pipeline': [{'type': 'LoadImageFromFile'}, 
                        {'type': 'Resize', 'scale': (2048, 560), 'keep_ratio': True}, 
                        {'type': 'LoadAnnotations'}, {'type': 'PackSegInputs'}]}

dataloader_cfg = dict(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    persistent_workers=persistent_workers
                )


seed = 976513493 # we can have a seed

data_loader = Runner.build_dataloader(dataloader_cfg, seed=seed)
first_batch = next(iter(data_loader))
print(f'First batch: {first_batch}')

input_image = first_batch['inputs'][0] # first image in the batch
gt_mask = first_batch['data_samples'][0].gt_sem_seg.data # first mask in the batch BUT it is in the original size