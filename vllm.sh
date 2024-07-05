#!/usr/bin/env bash

IMAGE_VERSION=latest
IMAGE_NAME=vllm-zj/dpo-72b
CONTAINER_NAME=vllm72b-zj-deploy
MODEL_DIR=/mnt/geogpt-gpfs/ali/models/Qwen/Qwen-geo/geogpt-72b-0412
DEVICES='"device=0,1,2,3"'

start() {
    # docker start command
    echo "start run docker..."
    docker run -d --name ${CONTAINER_NAME} \
        --log-opt max-size=30m \
        --log-opt max-file=3 \
        --gpus=${DEVICES} \
        --shm-size=16g \
        --network host \
        -e LANG="C.UTF-8" \
        -e LC_ALL="C.UTF-8" \
        -e MODEL_PATH=${MODEL_DIR} \
        -e MODEL_TYPE="Qwen2" \
        -e PORT=18192 \
        -v ${MODEL_DIR}:${MODEL_DIR} \
        ${IMAGE_NAME}:${IMAGE_VERSION}
}

$1
