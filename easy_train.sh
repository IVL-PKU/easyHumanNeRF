#! /bin/bash

ROOT=$(cd "$(dirname "$0")"; pwd)"/"
VIDEO="liwei.mp4"
IMAGES="your/images/"  # please use absolute path
EXPERIMENT_NAME="demo01"
WORKSPACE="workspace/"
date

echo "person segmentation..."

META_PATH=$ROOT$WORKSPACE$EXPERIMENT_NAME"/"

echo "\033[31mRemoving  $META_PATH...\033[0m"
rm -r $META_PATH
mkdir -p $META_PATH"/masks"

cd "dataset/wild/"
rm "monocular"
ln -s $META_PATH "monocular" 
cd -

echo "\033[33mIntermediate results in $META_PATH\033[0m"
cd $META_PATH
ln -s $IMAGES "images"
cd -

MASK_PATH="priors/mask/yolov7/"
POSE_PATH="priors/pose/VIBE/"

cd $MASK_PATH

python seg.py --input $IMAGES --output $META_PATH"masks"

echo "Mask completely!"

echo "Handling pose..."

mkdir -p $META_PATH"pose"

cd $ROOT$POSE_PATH
python handle_metas_for_humannerf.py --vid_file $IMAGES --output_folder $META_PATH"pose" 

echo "Pose ready!"

echo "Handling metadata"

cd $ROOT"priors/"
python transform_camera.py --input $META_PATH"/pose/vibe_output.pkl" --output $META_PATH"metadata.json"

cd $ROOT"tools/prepare_wild/"
python prepare_dataset.py --cfg wild.yaml


echo "Metadata ready, starting HumanNeRF..."

cd $ROOT

python train.py --cfg configs/human_nerf/wild/monocular/adventure.yaml
