import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.filters import threshold_otsu as sk_otsu
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from adapt import get_method
from utils import segmentation_datasets
from utils.metrics import intersect_and_union, process_metrics
from utils.misc import set_global_seeds, save_configuration, aggregate_pred_patches
from utils.pamr import PAMR


def argparser():
    parser = argparse.ArgumentParser("Weight Average Test Time Adaptation of CLIP")

    # Directories
    parser.add_argument('--save_dir', type=str, default='save/', help='Path for saving base training weights and results')

    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--plot_loss', action='store_true', help='Plot the loss curve (averged over batches and seeds)')

    # Model
    parser.add_argument('--backbone', type=str, default='ViT-B/32', help='Model backbone to use')
    parser.add_argument('--arch', type=str, default='reduced', choices=('reduced', 'vanilla'), help='Model backbone to use')
    parser.add_argument('--attn_strategy', type=str, default='naclip', choices=('naclip', 'nonly', 'kk', 'csa', 'vanilla'), help='Model backbone to use')
    parser.add_argument('--gaussian_std', type=float, default=5., help='Model backbone to use')
    parser.add_argument('--vision_outputs', nargs='+', default=(-1,), type=int, help='The indices of the vision layers to use. The default is the last layer of the vision backbone')
    parser.add_argument('--class_extensions', action='store_true', help='Enable class extensions if it is defined in the dataset')



    # Dataset settings #TODO: if the resize and patch size is not compatible what should we do? => we should define only compatible values
    parser.add_argument('--data_dir', type=str, default='/export/livia/home/vision/Mnoori/old_home/data/segmentation/coco_stuff164k', help='Root directory for datasets')
    parser.add_argument('--init_resize', nargs='+', default=None, type=int, help='Initial resize of the image in the dataloader => the order does not matter because the image will be resized based on the ratio so (560, 448) is the same as (448, 560). If none, the original image size will be used (bs should be 1)')
    parser.add_argument('--patch_size', nargs='+', default=None, type=int, help='The size of the patch that will be extracted from the resized image (model input size)')
    parser.add_argument('--patch_stride', type=int, default=None, help='The stride of the patch extraction')

    parser.add_argument('--dataset', type=str, default='COCOStuffDataset', choices=('COCOStuffDataset', 'COCOObjectDataset', 'CityscapesDataset', 'PascalVOC20Dataset', 'PascalVOC21Dataset', 'PascalContext59Dataset', 'PascalContext60Dataset'), help='Dataset to use')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers for data loading')

    # Training settings
    parser.add_argument('--batch_size', '--batch-size', type=int, default=128, dest='batch_size', help='Batch size for adaptation (can use --batch_size or --batch-size)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--trials', default=3, type=int, help='Number of trials to repeat the experiments')
    parser.add_argument('--steps', default=10, type=int, help='Number of iterations for adaptation')


    # Evaluation settings
    parser.add_argument('--adapt', action='store_true', help='Enable adaptation')
    parser.add_argument('--interpolate', action='store_true', help='If set, the model will interpolate the final logits during adaptation (for evaluation, it will be always interpolated)')
    parser.add_argument('--use_pamr', action='store_true', help='Enable PAMR post-processing')

    # Corruptions settings
    parser.add_argument('--corruptions_list', nargs='+', default=None, type=str, help='List of corruptions to apply to the dataset (Cifar datasets)')

    # Method name
    parser.add_argument('--method', type=str, default='watt', help='Method to use for adaptation')

    # templates used
    parser.add_argument('--temp_dir', type=str, default="templates.yaml", help='path to the templates.yaml file')

    return parser

def add_method_specific_args(parser, method):
    '''
    Add method-specific arguments to the parser
    '''
    if method == 'watt':
        parser.add_argument('--temperature', type=int, default=100, help='The temperature for the softmax (loss)')
        parser.add_argument('--l', default=2, type=int, help='Number of adaptation iterations for each text embedding before weight averaging')
        parser.add_argument('--m', default=5, type=int, help='Number of repetitions of the adaptation and weight averaging process')

    elif method == 'tent':
        parser.add_argument('--average_type', type=str, default='loss', help='If we have different text templates, how to average them (loss-level or text-level)')
        pass
    
    elif method == 'mtl':
        parser.add_argument('--average_type', type=str, default='loss', help='If we have different text templates, how to average them (loss-level or text-level)')
        parser.add_argument('--alpha_cls', type=float, default=0.0, help='Model backbone to use')

    elif method == 'clipartt':
        parser.add_argument('--K', default=3, type=int, help='Number of classes taken to build the area pseudo label')


    # Add other methods here
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return parser


def main(args):

    # Save the configuration settings
    save_configuration(args)

    # Start the timer
    start_time = time.time()

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # init PAMR 
    if args.use_pamr:
        pamr = PAMR(num_iter=10, dilations=(8, 16)).to(device)

    all_results_path = os.path.join(args.save_dir, "results.txt")
    os.makedirs(os.path.dirname(all_results_path), exist_ok=True)

    all_results = dict()
    headers = "mIoU, mDice, mAcc"
    for c_idx, corruption in enumerate(args.corruptions_list):
        data_loader, org_classes = segmentation_datasets.prepare_data(args.dataset, args.data_dir, args.init_resize,
                                                                  args.patch_size, args.patch_stride, corruption=corruption, 
                                                                  batch_size=args.batch_size, num_workers=args.workers)
        
        # Check if the extensions of classes should be used
        if args.class_extensions and data_loader.dataset.class_extensions is not None:
            ext_classes = data_loader.dataset.class_extensions
            args.classes = ext_classes
            print(f"\n+++ Using class extensions")
            print(f"+++ The number of classes [no extension]: {len(org_classes)}")
            print(f"+++ The number of classes after extension:  {len(ext_classes)}")

        else:
            args.classes = org_classes
            print(f"\n+++ The number of classes [no extension]: {len(org_classes)}")

        num_org_classes = len(org_classes)
        ignore_index = data_loader.dataset.ignore_index # the index of the ignore label in the segmentation map

        # Setting up the model and the method
        adapt_method = get_method(args, device)

        # print information about the vision layers used
        print(f"+++ The output layers from vision encoder that will be used: {args.vision_outputs}")

        # Results path
        c_results_path = os.path.join(args.save_dir, f"{c_idx:02}_{corruption}", "results.txt")
        os.makedirs(os.path.dirname(c_results_path), exist_ok=True)

        miou_seeds = []
        dice_seeds = []
        acc_seeds = []

        loss_seed_report = []

        for t in range(args.trials):
            results = []
            loss_batch_report = []
            for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
                inputs = data['img_patches'] # [real_bs, 3, 224, 224]
                labels = data['gt_patches']  # [real_bs, 1, 224, 224]
                original_gts = data['gt'] # a list of the original segmentation maps (after init_Rezize and before patchifying) for each image in the batch

                patch_grid_shape = data['meta']['patch_grid_shape'] # the shape of patch grids based on the original image, ex for imagin that it is (4, 3) for the first image, it means that the image is divided into 4 rows and 3 columns of patches
                image_shapes = data['meta']['img_shape'] # the shape of the original image before patchifying and after resizing to init_resize, ex for the first image it is (560, 448)
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # reset the model before adapting to a new batch
                adapt_method.reset()
                
                # perform adaptation
                if args.adapt:
                    loss_iter_report = adapt_method.adapt(inputs, args.classes, vision_outputs=args.vision_outputs)
                    loss_batch_report.append(loss_iter_report)

                # perform evaluation 
                patch_preds = adapt_method.evaluate(inputs, args.classes, vision_outputs=args.vision_outputs)

                # aggregate the predictions to construct the final segmentation map for each image in the batch
                if args.init_resize:
                    reconstructed_preds = aggregate_pred_patches(patch_preds, patch_grid_shape, image_shapes, args.patch_size, args.patch_stride)
                else:
                    reconstructed_preds = patch_preds

                
                # calculate the metrics for each image in the batch (since the images may have different sizes)
                for idx, (pd, gt) in enumerate(zip(reconstructed_preds, original_gts)):

                    # apply PAMR 
                    if args.use_pamr: #FIX PAMR FOR real sizes
                        org_img = data['img'][idx].to(pd.device)
                        pd = pamr(org_img.unsqueeze(0), pd.unsqueeze(0))[0]

                    # get the predictions
                    pd = pd.softmax(dim=0) # [num_org_classes, H, W]

                    # fix the extensions indices
                    if args.class_extensions and data_loader.dataset.class_extensions is not None:
                        ext_to_real_cls_indx = torch.Tensor(data_loader.dataset.extentions_to_real_class_idx).to(torch.int64).to(device)
                        num_cls, num_queries = max(ext_to_real_cls_indx) + 1, len(ext_to_real_cls_indx)
                        ext_to_real_cls_indx = nn.functional.one_hot(ext_to_real_cls_indx)
                        ext_to_real_cls_indx = ext_to_real_cls_indx.T.view(num_cls, num_queries, 1, 1)
                        pd = pd.unsqueeze(0)
                        pd = (pd * ext_to_real_cls_indx).max(1)[0]


                    pd = pd.argmax(dim=0)  # [H, W]
                    pd = pd.to(gt.device)  

                    # get the ground truth
                    gt = gt[0]             # [H, W]
                    # metric calculation
                    results.append(intersect_and_union(pd, gt, num_org_classes, ignore_index))
               
            
            # Convert the batch report to a numpy array for easier averaging
            loss_batch_report = np.array(loss_batch_report)

            # Average loss over batches for each iteration
            avg_loss_per_iter = np.mean(loss_batch_report, axis=0)  # Shape: [10] (for 10 iterations)
            loss_seed_report.append(avg_loss_per_iter)

            
            metrics = process_metrics(results, org_classes)
            miou_seeds.append(metrics['mIoU'])
            dice_seeds.append(metrics['mDice'])
            acc_seeds.append(metrics['mAcc'])
            print(f"Results for corruption: {corruption}, trial: {t}, mIoU:  {metrics['mIoU']}, mDice:  {metrics['mDice']}, mAcc: {metrics['mAcc']}")


            ### saving the weights if self.weights_track list is not empty
            if adapt_method.model.weights_track:
                weights_path = os.path.join(args.save_dir, "weights")
                
                weights = adapt_method.model.weights_track
                weights = np.hstack(weights)
                os.makedirs(weights_path, exist_ok=True)
                
                # save to a file
                np.save(os.path.join(weights_path, f"{corruption}_s{t}.npy"), np.array(weights))

                # plot and save the mean and std of weights across the layers
                weights_mean = np.mean(weights, axis=1)
                weights_std = np.std(weights, axis=1)
                plt.figure()
                plt.errorbar(range(len(weights_mean)), weights_mean, yerr=weights_std, fmt='o')
                plt.xlabel('Layer')
                plt.ylabel('Weight')
                plt.title(f'Mean and Std of Weights for {corruption}')
                plt.savefig(os.path.join(weights_path, f"{corruption}_s{t}.png"))
                plt.close()

                # reset the weights_track list
                adapt_method.model.weights_track = []

                # debug
                # top_k_indices = np.argsort(weights, axis=0)[-5:, :]
                # print(top_k_indices)
                    

        
        miou_mean, miou_std = np.array(miou_seeds).mean(), np.array(miou_seeds).std()
        dice_mean, dice_std = np.array(dice_seeds).mean(), np.array(dice_seeds).std()
        acc_mean, acc_std = np.array(acc_seeds).mean(), np.array(acc_seeds).std()

        print(f"mIoU:  {miou_mean:.2f},{miou_std:.2f}")
        print(f"mDice: {dice_mean:.2f},{dice_std:.2f}")
        print(f"mAcc:  {acc_mean:.2f},{acc_std:.2f}")

        c_results_print = f"{miou_mean:.2f} +/- {miou_std:.2f}, {dice_mean:.2f} +/- {dice_std:.2f}, {acc_mean:.2f} +/- {acc_std:.2f}"
        with open(c_results_path, 'w') as f:        
            f.write(headers + "\n")
            f.write(c_results_print)    

        all_results[corruption] = c_results_print

        # Convert the seed report to a numpy array and average over trials (seeds)
        loss_seed_report = np.array(loss_seed_report)
        avg_loss_over_seeds = np.mean(loss_seed_report, axis=0)  # Shape: [10] (averaged over seeds)

        if args.plot_loss and args.adapt:
            # Plot the averaged loss for this corruption
            plt.figure()
            plt.plot(range(1, len(avg_loss_over_seeds)+1), avg_loss_over_seeds)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title(f'Average Loss per Iteration for {corruption}')
            
            # Save the plot in the specified directory
            save_path = os.path.join(args.save_dir, f'loss_{corruption}.png')
            plt.savefig(save_path)
            plt.close()

    total_duration = time.time() - start_time
    mean_duration_per_seed = total_duration / args.trials
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


    with open(all_results_path, 'w') as f:
        f.write(headers + "\n")
        for corruption, results in all_results.items():
            f.write(f"{corruption}, {results}\n")
        f.write(f"\nGPU: {gpu_info}\n")
        f.write(f"Total Duration (s): {total_duration:.2f}\n")
        f.write(f"Mean Duration per Seed (s): {mean_duration_per_seed:.2f}\n")

if __name__ == "__main__":
    # Initial argument parsing to get the method
    initial_parser = argparser()
    initial_args, _ = initial_parser.parse_known_args()

    # Create a new parser with method-specific arguments
    parser = argparser()
    parser = add_method_specific_args(parser, initial_args.method)
    args = parser.parse_args()

    # Set the global random seed for reproducibility
    set_global_seeds(args.seed)

    main(args)
