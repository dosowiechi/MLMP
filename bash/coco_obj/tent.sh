# GPU Configuration
GPU_ID=0

# Dataset Configuration
DATASET=COCOObjectDataset
DATA_DIR=".data/coco_object/"
INIT_RESIZE="224 224"
ALL_CORRUPTIONS="original"
WORKERS=4

# Method and OVSS Model Configuration
METHOD="tent"
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

