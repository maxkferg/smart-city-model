"""
Test each of the modules
Modules can be selected using command line arguments
Runs the test.py file in module directory
"""
import sys
import numpy as np
import tensorflow as tf

print("Using numpys version ==", np.__version__)
print("Using tensorflow version ==", tf.__version__)
print("Running module tests")

from modules.vision.train_ssd_network import train
train()


"""

python3 tf_convert_data.py \
--dataset_name=database \
--dataset_dir=1500327953 \
--output_name=database_train \
--output_dir=/Users/maxferguson/Data/database


# =========================================================================== #
# Database training
# =========================================================================== #
DATASET_NAME=database
TRAIN_DIR=./modules/ssd/logs/ssd_300_database
DATASET_DIR=/Users/maxferguson/Data/database
CHECKPOINT_PATH=./modules/ssd/checkpoints/ssd_model.ckpt

python3 train.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_name=${DATASET_NAME} \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.000001 \
    --batch_size=32 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \



"""