#!/usr/bin/env bash

IMAGE_VERSION=23.08-py3
IMAGE_NAME=nvcr.io/nvidia/pytorch
CONTAINER_NAME=vllm72b-zj
MODEL_DIR=/mnt/geogpt/ali/models/Qwen/geo_72B_sft_ckpt_mp8_pp4_hf-2
CODE_DIR=/home/chaizehua/code/vllm
DEVICES='"device=0,1,2,3,4,5,6,7"'

start() {
    # docker start command
    echo "start run docker..."
    docker run -it --name ${CONTAINER_NAME} \
        --log-opt max-size=30m \
        --log-opt max-file=3 \
        --gpus=${DEVICES} \
        --shm-size=16g \
        --network host \
        -e LANG="C.UTF-8" \
        -e LC_ALL="C.UTF-8" \
        -e MODEL_PATH=${MODEL_DIR} \
        -e MODEL_TYPE="Qwen" \
        -v ${MODEL_DIR}:${MODEL_DIR} \
        -v ${CODE_DIR}:${CODE_DIR} \
        ${IMAGE_NAME}:${IMAGE_VERSION}
}

start