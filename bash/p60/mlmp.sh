# GPU Configuration
GPU_ID=0

# Dataset Configuration
DATASET=PascalContext60Dataset
DATA_DIR=".data/VOC2010/"
INIT_RESIZE="224 224"
ALL_CORRUPTIONS="original gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
WORKERS=4

# Method and OVSS Model Configuration
METHOD="mlmp"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"
ALPHA_CLS=1.0
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters
BATCH_SIZE=2
LR=0.001
STEPS=10
TRIALS=3

# Output
SAVE_DIR=".save/${DATASET}/${METHOD}/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
                        --adapt \
                        --method $METHOD \
                        --prompt_dir $PROMPT_DIR \
                        --vision_outputs $OUT_VISION \
                        --alpha_cls $ALPHA_CLS \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        \
                        --save_dir $SAVE_DIR \
                        --data_dir $DATA_DIR \
                        --dataset $DATASET \
                        --workers $WORKERS \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $ALL_CORRUPTIONS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch-size $BATCH_SIZE \
                        --trials $TRIALS \
                        --seed 0 \
                        \
                        --plot_loss \
                        --class_extensions

