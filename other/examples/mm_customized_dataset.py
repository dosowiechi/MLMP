
import sys
import os
main_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, main_project_dir)

import utils.mm_transforms as mm_transforms
from utils.misc import custom_collate

from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import Resize

from mmengine.registry import TRANSFORMS
from mmengine.registry import DATASETS

from mmseg.datasets import CityscapesDataset, COCOStuffDataset
from mmseg.datasets.transforms.loading import LoadAnnotations
from mmseg.datasets.transforms.formatting import PackSegInputs

from torch.utils.data import DataLoader


# Register the modules with TRANSFORMS
TRANSFORMS.register_module(module=LoadImageFromFile, force=True)
TRANSFORMS.register_module(module=Resize, force=True)
TRANSFORMS.register_module(module=LoadAnnotations, force=True)
TRANSFORMS.register_module(module=PackSegInputs, force=True)
DATASETS.register_module(module=CityscapesDataset, force=True)
DATASETS.register_module(module=COCOStuffDataset, force=True)



batch_size = 2 # numnfer of loaded images
resize = (560, 448) # the size of the image after resizing => it can be vertical or horizontal depending on the image so => (560, 448) or (448, 560)
patch_size = (224, 224) # the size of the patch that will be extracted from the resized image
patch_stride = 112 # the stride of the patch extraction

num_workers = 4

mm_dataset_cfg =  { 
    'type': 'COCOStuffDataset', 
    'data_root': '/export/livia/home/vision/Mnoori/old_home/data/segmentation/coco_stuff164k', 
    'data_prefix': {'img_path': 'images/val2017', 'seg_map_path': 'annotations/val2017'}, 
    'pipeline': [{'type': 'LoadImageFromFile'}, 
                {'type': 'LoadAnnotations'}, 
                {'type': 'ResizeAndPatchify', 'resize': resize, 'patch_size': patch_size, 'patch_stride': patch_stride},
                {'type': 'ToTensorAndNormalize', 'mean': [122.7709, 116.7460, 104.0937], 'std': [68.5005, 66.6322, 70.3232]},
                ]
    }


# data_loader = Runner.build_dataloader(dataloader_cfg, seed=seed) 
dataset = DATASETS.build(mm_dataset_cfg)

# Note the final batch size is batch_size * num_patches
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=custom_collate, persistent_workers=True)

data = next(iter(dataloader))
print(f'Input Patches: {data["img_patches"].shape}')
print(f'Patch GTs:     {data["gt_patches"].shape}')
print(data["gt_patches"][0])