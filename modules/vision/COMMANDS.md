# =========================================================================== #
# Dataset convert...
# =========================================================================== #
rm events* graph* model* checkpoint
mv events* graph* model* checkpoint ./log
tensorboard --logdir=logs/ssd_300_voc

DATASET_DIR=/Users/maxferguson/data/kitti/training
OUTPUT_DIR=/Users/maxferguson/data
python tf_convert_data.py \
    --dataset_name=kitti \
    --dataset_dir=${DATASET_DIR} \
    --output_name=kitti_train \
    --output_dir=${OUTPUT_DIR}

# =========================================================================== #
# VGG-based SSD network [MAX July 4th]
# =========================================================================== #

DATASET_DIR=/Users/maxkferg/Onedrive/Stanford/PhD/data/VOC2007/
OUTPUT_DIR=/Users/maxkferg/Onedrive/Stanford/PhD/data/VOC2007-SSD/
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=pascalvoc_2007_train \
    --output_dir=${OUTPUT_DIR}


CAFFE_MODEL=/media/paul/DataExt4/PascalVOC/training/ckpts/SSD_300x300_ft/ssd_300_vgg.caffemodel
python caffe_to_tensorflow.py \
    --model_name=ssd_300_vgg_caffe \
    --num_classes=21 \
    --caffemodel_path=${CAFFE_MODEL}

# =========================================================================== #
# VGG-based SSD network [MAX July 4th]
# =========================================================================== #
DATASET_NAME=database
TRAIN_DIR=./logs/ssd_300_database
CHECKPOINT_PATH=./checkpoints/ssd_300_database.ckpt

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
    --learning_rate=0.00005 \
    --learning_rate_decay_factor=0.999
    --batch_size=32 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \



# =========================================================================== #
# VGG-based SSD network [MAX July 4th]
# =========================================================================== #
DATASET_DIR=/Users/maxkferg/Onedrive/Stanford/PhD/data/VOC2007-SSD/
TRAIN_DIR=./logs/ssd_300_voc
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt

python3 train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --dataset_name=pascalvoc_2007 \
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



# =========================================================================== #
# VGG-based SSD network
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/KITTI/dataset
TRAIN_DIR=./logs/ssd_300_kitti_5
CHECKPOINT_PATH=/media/paul/DataExt4/PascalVOC/training/ckpts/SSD_300x300_ft/ssd_300_vgg.ckpt
CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
python train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --dataset_name=kitti \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.0005 \
    --optimizer=rmsprop \
    --learning_rate=0.000001 \
    --batch_size=32

    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \

# =========================================================================== #
# Inception v3
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
DATASET_DIR=../datasets/ImageNet
TRAIN_DIR=./logs/inception_v3
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
CHECKPOINT_PATH=./checkpoints/inception_v3.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=4


CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/logs
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v3


# =========================================================================== #
# VGG 16 and 19
# =========================================================================== #
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/vgg_19.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --labels_offset=1 \
    --dataset_split_name=validation \
    --model_name=vgg_19


CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/vgg_16.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --labels_offset=1 \
    --dataset_split_name=validation \
    --model_name=vgg_16


# =========================================================================== #
# Xception
# =========================================================================== #
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=./logs/xception
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=xception \
    --labels_offset=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=600 \
    --save_interval_secs=600 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.0001 \
    --batch_size=32

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=xception \
    --labels_offset=1 \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=1


CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
CHECKPOINT_PATH=./logs/xception
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --labels_offset=1 \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=xception \
    --max_num_batches=10


CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.h5
python ckpt_keras_to_tensorflow.py \
    --model_name=xception_keras \
    --num_classes=1000 \
    --checkpoint_path=${CHECKPOINT_PATH}


# =========================================================================== #
# Dception
# =========================================================================== #
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=./logs/dception
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=dception \
    --labels_offset=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=32

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=dception \
    --labels_offset=1 \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=1


CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
CHECKPOINT_PATH=./logs/dception
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --labels_offset=1 \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=dception
