import os
import yaml
import torch
import random
import subprocess

import numpy as np
from datetime import datetime


def set_global_seeds(seed_value=42):
    """Set random seeds for reproducibility across various libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_git_commit_hash():
    """
    Get the current Git commit hash.
    """
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
        return commit_hash
    except Exception as e:
        print(f"Error retrieving Git commit hash: {e}")
        return "Unknown"

def save_configuration(args, config_file="configurations.txt", cmd_file="cmd.sh"):
    """
    Save configuration parameters, Git commit hash, and the current date to a file.
    Also save the command line used to run the script.
    """
    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Retrieve the Git commit hash
    commit_hash = get_git_commit_hash()
    
    # Get the current date and time
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save configurations to a file
    config_filepath = os.path.join(args.save_dir, config_file)
    print("\nConfigurations +++++++++++++++++++++++++")
    print("----------------------------------------")

    with open(config_filepath, 'w') as config_f:
        # Save the date
        config_f.write(f"date: {current_date}\n")
        print(f"       date: {current_date}")
        
        # Save Git commit hash
        config_f.write(f"git_commit_hash: {commit_hash}\n")
        print(f"       git_commit_hash: {commit_hash}")
        
        # Save other configurations
        for arg in vars(args):
            value = getattr(args, arg)
            config_f.write(f"{arg}: {value}\n")
            print(f"       {arg}: {value}")

    # Save the command to a file
    cmd_filepath = os.path.join(args.save_dir, cmd_file)
    arg_dict = vars(args)
    cmd = "python main_segmentation.py"
    for key, value in arg_dict.items():
        # Use the exact argument name with underscores
        formatted_key = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd += f" {formatted_key}"
        elif isinstance(value, list) or isinstance(value, tuple):
            # Join list or tuple items with spaces
            cmd += f" {formatted_key} {' '.join(map(str, value))}"
        elif value is not None:
            cmd += f" {formatted_key} {value}"

    with open(cmd_filepath, "w") as cmd_f:
        cmd_f.write(cmd + "\n")
    print(f"Configurations & Command saved!")
    print("----------------------------------------")




def load_prompts_from_yaml(file_path='prompts.yaml'):
    """Load prompt templates from a YAML file."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['prompt_templates'] 


def save_checkpoint(state, is_best, args):
    torch.save(state, args.save + args.dataset + '_' + args.model + '.pth')
    if is_best:
            torch.save(state, args.save + args.dataset + '_' + args.model + '_torch_best.pth')


def print_clip_parameters(model):
    """
    Print the total and learnable parameters (requires_grad=True) for each module in a CLIP model
    and the overall summary.

    Args:
        model (torch.nn.Module): The PyTorch model.
    """
    modules = {
        "model.visual": model.visual,
        "model.transformer": model.transformer,
        "model.ln_final": model.ln_final,
        "model.token_embedding": model.token_embedding
    }
    
    print("\nModel Parameters Summary +++++++++++++++++++")
    
    total_params = 0
    learnable_params = 0

    # Print parameters for each module
    for name, module in modules.items():
        module_total = sum(p.numel() for p in module.parameters())
        module_learnable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += module_total
        learnable_params += module_learnable
        print(f"{name:25}: Total = {module_total:,}, Learnable = {module_learnable:,}")

    # Print overall summary
    print("----------------------------------------")
    print(f"Total Parameters      : {total_params:,}")
    print(f"Learnable Parameters  : {learnable_params:,}")
    print("----------------------------------------")


def print_optimizer_parameters(optimizer, model):
    """
    Print the total and learnable parameters passed to the optimizer,
    grouped by each module of the CLIP model.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.
        model (torch.nn.Module): The PyTorch model.
    """
    # Define the modules to analyze
    modules = {
        "model.visual": model.visual,
        "model.transformer": model.transformer,
        "model.ln_final": model.ln_final,
        "model.token_embedding": model.token_embedding
    }
    
    # Collect parameters in the optimizer
    optimizer_params = {id(p): p for group in optimizer.param_groups for p in group['params']}
    
    print("\nOptimizer Parameters by Module +++++++++++++")
    total_optimizer_params = 0

    # Count parameters for each module
    for name, module in modules.items():
        module_params = sum(
            p.numel() for p in module.parameters() if id(p) in optimizer_params
        )
        total_optimizer_params += module_params
        print(f"{name:25}: Parameters in Optimizer = {module_params:,}")

    # Print the total
    print("---------------------------------------------")
    print(f"Total Parameters in Optimizer : {total_optimizer_params:,}")



def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = list(), list()
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices





def custom_collate(data):
    # Extract each key from the dictionary
    imgs_list = [item['img'] for item in data] # we cannot stack the images as they have different sizes
    gt_list = [item['gt_seg_map'] for item in data] # we cannot stack the images as they have different sizes
    


    # For patches and gt_seg_map_patches, stack on axis=0 for each sample
    img_patches_list= [item['patches'] for item in data]
    gt_patches_list= [item['gt_seg_map_patches'] for item in data]
    
    # Stack the list of stacked patches along the new batch axis
    img_patches_batch= torch.cat(img_patches_list, axis=0)
    gt_patches_batch= torch.cat(gt_patches_list, axis=0)
    
    # Collect meta information as a dictionary
    meta = {
        'img_path': [item['meta']['img_path'] for item in data],
        'seg_map_path': [item['meta']['seg_map_path'] for item in data],
        'ori_shape': [item['meta']['ori_shape'] for item in data],
        'img_shape': [item['meta']['img_shape'] for item in data],
        'patch_shape': [item['meta']['patch_shape'] for item in data],
        'scale_factor': [item['meta']['scale_factor'] for item in data],
        'patch_grid_shape': [item['meta']['patch_grid_shape'] for item in data],
    }

    # Return as a single dictionary
    return {
        'img': imgs_list,
        'gt': gt_list,
        'img_patches': img_patches_batch,
        'gt_patches': gt_patches_batch,
        'meta': meta
    }




def aggregate_pred_patches(pred, patch_grid_shapes, img_shapes, patch_size=(224, 224), patch_stride=112):
    """
    Aggregates per-class score patches back to the original image size for each image in the batch.
    
    Args:
        pred (torch.Tensor): Predictions of shape (batch_size * num_patches, num_classes, 224, 224).
        patch_grid_shapes (list of tuple): List of grid dimensions (num_patches_y, num_patches_x) for each image.
        img_shapes (list of tuple): List of original image shapes (height, width) for each image.
        patch_size (tuple): Size of each patch.
        patch_stride (int): Stride between patches for overlapping areas.
        
    Returns:
        list of torch.Tensor: List of reconstructed predictions, each of shape (num_classes, img_height, img_width).
    """
    batch_size = len(patch_grid_shapes)
    num_classes = pred.shape[1]  # Number of classes in the segmentation task
    reconstructed_preds = []
    
    patch_idx = 0  # Tracks global patch index for the batch
    
    for b in range(batch_size):
        num_patches_y, num_patches_x = patch_grid_shapes[b]
        img_height, img_width = img_shapes[b]
        
        # Initialize tensors for the reconstructed per-class score image and overlap counts
        reconstructed_pred = torch.zeros((num_classes, img_height, img_width), device=pred.device)
        overlap_count = torch.zeros((1, img_height, img_width), device=pred.device)
        
        # Place patches according to their positions
        for i in range(num_patches_y):
            for j in range(num_patches_x):
                # Calculate patch placement coordinates
                start_y = i * patch_stride
                start_x = j * patch_stride
                end_y = start_y + patch_size[0]
                end_x = start_x + patch_size[1]
                
                # Add per-class scores of patch to the reconstructed image
                reconstructed_pred[:, start_y:end_y, start_x:end_x] += pred[patch_idx]
                overlap_count[:, start_y:end_y, start_x:end_x] += 1
                patch_idx += 1
        
        # Average overlapping areas for each class
        reconstructed_pred /= overlap_count
        reconstructed_preds.append(reconstructed_pred)
    
    return reconstructed_preds  # Returns a list of per-class score tensors for each image

