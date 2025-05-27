# Standard
import os, time, argparse

# Third-party
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local
from adapt import get_method
from utils import segmentation_datasets
from utils.metrics import intersect_and_union, process_metrics
from utils.misc import set_global_seeds, save_configuration, aggregate_pred_patches


"""TODO List:
- end of main_segmentation is necessary?


- datasets
    we can have download datasets script? not sure
    but we can have a dataset.md
- repo:
    - add a section (supported methods=> list them and add reference to each of them)
    - we can talk about how to perform all methods (including No Adapt)
    - in acknowledgements, we can say athat we modified the original CLIP code to "ovss/clip/model.py" to be able to perform segmentation 
"""



def argparser():
    parser = argparse.ArgumentParser(
        description="Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation"
    )
    
    # ----------------------------------------
    # I/O Directories
    # ----------------------------------------
    parser.add_argument(
        '--save_dir',
        type=str,
        default='save/',
        help='Directory to save model weights and results'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='.data/',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--prompt_dir',
        type=str,
        default='',
        help='Path to the YAML file containing prompt templates'
    )
    
    # ----------------------------------------
    # Dataset Settings
    # ----------------------------------------
    parser.add_argument(
        '--dataset',
        type=str,
        default='COCOStuffDataset',
        choices=(
            'COCOStuffDataset', 'COCOObjectDataset', 'CityscapesDataset',
            'PascalVOC20Dataset', 'PascalVOC21Dataset',
            'PascalContext59Dataset', 'PascalContext60Dataset'
        ),
        help='Which dataset to load'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of data-loading workers'
    )
    parser.add_argument(
        '--init_resize',
        nargs='+',
        type=int,
        default=None,
        help=(
            'Resize images before patch extraction. '
            'Order doesn’t matter (e.g., (560,448) same as (448,560)). '
            'If None, use original size (batch_size must be 1).'
        )
    )
    parser.add_argument(
        '--patch_size',
        nargs='+',
        type=int,
        default=None,
        help='Size of each image patch after resize (model input size)'
    )
    parser.add_argument(
        '--patch_stride',
        type=int,
        default=None,
        help='Stride for extracting patches'
    )
    parser.add_argument(
        '--corruptions_list',
        nargs='+',
        type=str,
        default=None,
        help='List of corruptions to apply for robustness (e.g., gaussian, motion_blur)'
    )
    
    # ----------------------------------------
    # Model Settings
    # ----------------------------------------
    parser.add_argument(
        '--ovss_type',
        type=str,
        default='ncalip',
        help='Open-Vocabulary Semantic Segmentation type (e.g., nacalip, clip, clip, etc.)'
    )
    parser.add_argument(
        '--ovss_backbone',
        type=str,
        default='ViT-B/32',
        help='CLIP vision backbone (e.g., ViT-B/32, ViT-L/14)'
    )
    parser.add_argument(
        '--class_extensions',
        action='store_true',
        help='Enable dataset-specific class extensions if available'
    )
    
    # ----------------------------------------
    # Adaptation / Training Settings
    # ----------------------------------------
    parser.add_argument(
        '--adapt',
        action='store_true',
        help='Enable test-time adaptation'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='tent',
        help='Adaptation method name (e.g., mlmp watt, tent)'
    )
    parser.add_argument(
        '--batch_size', '--batch-size',
        type=int,
        default=128,
        dest='batch_size',
        help='Batch size for adaptation'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate for adaptation optimizer'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=10,
        help='Number of adaptation iterations per batch'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=3,
        help='Number of experimental repetitions'
    )
    
    # ----------------------------------------
    # Debug / Misc
    # ----------------------------------------
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--plot_loss',
        action='store_true',
        help='Plot the loss curve (averaged over batches and seeds)'
    )
    parser.add_argument(
        '--runtime_calculation',
        action='store_true',
        help='Calculate the runtime of adaptation and evaluation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    return parser

def add_method_specific_args(parser, method):
    '''
    Add method-specific arguments to the parser
    '''
    if method == 'mlmp':
        parser.add_argument(
            '--vision_outputs',
            nargs='+',
            type=int,
            default=(-1,),
            help='Indices of vision layers to extract outputs from'
        )
        parser.add_argument(
            '--prompt_integration',
             type=str, default='loss', 
             help='If we have different prompt templates, how to integrate them (loss-level or text-level). MLMP uses loss-level integration by default.'
             )
        parser.add_argument(
            '--alpha_cls', 
            type=float, 
            default=1.0, 
            help='Weight for the classification loss in MLMP'
            )
    
    elif method == 'watt':
        parser.add_argument(
            '--watt_l', 
            default=2, 
            type=int, 
            help='Number of adaptation iterations for each text embedding before weight averaging'
            )
        parser.add_argument('--watt_m', 
            default=5, 
            type=int, 
            help='Number of repetitions of the adaptation and weight averaging process'
            )

    elif method == 'clipartt':
        parser.add_argument(
            '--clipartt_k', 
            default=3, 
            type=int, 
            help='Number of classes taken to build the area pseudo label'
            )

    elif method == 'tpt':
        parser.add_argument(
                '--n_ctx', 
                default=4, 
                type=int,
            )
    
    return parser


def main(args):

    # Save the configuration settings
    save_configuration(args)

    # Start the timer
    start_time = time.time()

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the save directory if it doesn't exist
    all_results_path = os.path.join(args.save_dir, "results.txt")
    os.makedirs(os.path.dirname(all_results_path), exist_ok=True)

    # create necessary variables
    all_results = dict()
    headers = "mIoU, mDice, mAcc"
    adapt_time_all_corr = []
    eval_time_all_corr = []
    
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

                if args.debug and batch_idx == 10: 
                    break

                inputs = data['img_patches'] 
                labels = data['gt_patches']  
                original_gts = data['gt'] 

                patch_grid_shape = data['meta']['patch_grid_shape'] 
                image_shapes = data['meta']['img_shape']
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # reset the model before adapting to a new batch
                adapt_method.reset()
                
                # perform adaptation
                if args.adapt:
                    loss_iter_report = adapt_method.adapt(inputs)
                    loss_batch_report.append(loss_iter_report)

                # perform evaluation 
                with torch.no_grad():
                    patch_preds = adapt_method.evaluate(inputs)

                # aggregate the predictions to construct the final segmentation map for each image in the batch
                if args.init_resize:
                    reconstructed_preds = aggregate_pred_patches(patch_preds, patch_grid_shape, image_shapes, args.patch_size, args.patch_stride)
                else:
                    reconstructed_preds = patch_preds

                
                # calculate the metrics for each image in the batch (since the images may have different sizes)
                for idx, (pd, gt) in enumerate(zip(reconstructed_preds, original_gts)):

                    # get the predictions
                    pd = pd.softmax(dim=0) # [num_org_classes, H, W]

                    # fix the extensions indices
                    if args.class_extensions and data_loader.dataset.class_extensions is not None:
                        ext_to_real_cls_indx = torch.Tensor(data_loader.dataset.extentions_to_real_class_idx).to(torch.int64).to(device)
                        num_cls, num_queries = max(ext_to_real_cls_indx) + 1, len(ext_to_real_cls_indx)
                        ext_to_real_cls_indx = torch.nn.functional.one_hot(ext_to_real_cls_indx)
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


            # Saving the weights if self.weights_track list is not empty
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

        # if the runtime calculation is enabled, we will have access to adapt_method.adapt_times and adapt_method.eval_times (each one contains a list of times)
        if args.runtime_calculation:
            if args.adapt:
                mean_adapt_time = np.mean(adapt_method.adapt_times[20:])
                std_adapt_time = np.std(adapt_method.adapt_times[20:])
            else:
                mean_adapt_time = 0
                std_adapt_time = 0
            
            mean_eval_time = np.mean(adapt_method.eval_times[20:])
            std_eval_time = np.std(adapt_method.eval_times[20:])

            mean_total_time = mean_adapt_time + mean_eval_time

            run_time_txt = f"{corruption}, {mean_adapt_time:0.3f} +/- {std_adapt_time:0.3f}, {mean_eval_time:0.3f} +/- {std_eval_time:0.3f}, {mean_total_time:0.3f}"
            print(run_time_txt)
            
            runtime_save_dir = os.path.join(args.save_dir, "runtime.txt")
            with open(runtime_save_dir, 'a+') as f:
                f.write(run_time_txt + "\n")

            adapt_time_all_corr.append(mean_adapt_time)
            eval_time_all_corr.append(mean_eval_time)

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

    # Run the main function with the parsed arguments
    main(args)
