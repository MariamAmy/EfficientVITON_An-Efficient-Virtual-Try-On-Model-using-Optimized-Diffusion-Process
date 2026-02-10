#!/bin/bash

# Start Directory
BASE_DIR="/home/user/stableVTON"
PREPROCESSING_DIR="$BASE_DIR/preprocessing"
DATA_DIR="$BASE_DIR/StableVITON/DATA/zalando-hd-resized/inference"

mkdir "$DATA_DIR/openpose_json"
mkdir "$DATA_DIR/openpose_img"
mkdir "$DATA_DIR/image-parse-v3"

# Record start time
start_time=$(date +%s)

##### OpenPose #####
cd "$PREPROCESSING_DIR/openpose"
source ./openpose_env/bin/activate

cd "./openpose"
./build/examples/openpose/openpose.bin --image_dir "$DATA_DIR/image" \
--write_json "$DATA_DIR/openpose_json" \
--write_images "$DATA_DIR/openpose_img" \
--display 0 --disable_blending

# Deactivate OpenPose environment
deactivate

# ##### DensePose #####
cd "$PREPROCESSING_DIR/densepose"
source ./densepose_env/bin/activate

cd ./detectron2/projects/DensePose
python3 apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
"$DATA_DIR/image" dp_segm -v

# ##### LIP Parsing #####
cd "$PREPROCESSING_DIR/LIP Parsing/Self-Correction-Human-Parsing"
python3 simple_extractor.py --dataset 'lip' --model-restore 'checkpoints/final.pth' \
--input-dir "$DATA_DIR/image" \
--output-dir "$DATA_DIR/image-parse-v3"

# ##### Agnostic + Agnostic Mask #####
cd "$PREPROCESSING_DIR"
python3 agnostic.py --data_path "$DATA_DIR/"

# ##### Parse Agnostic #####
python3 parse_agnostic.py --data_path "$DATA_DIR/"

# ##### GT Warp #####
python3 gt_warp.py --data_path "$DATA_DIR/"

# # Record end time before "Cloth Mask" process
end_time=$(date +%s)

# # Calculate and display total time taken
total_time=$((end_time - start_time))
echo "Total time taken for all processes (except Cloth Mask): $total_time seconds"

# ##### Cloth Mask #####
# python3 cloth_mask.py --data_path "$DATA_DIR/"
