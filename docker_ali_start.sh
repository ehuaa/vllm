#!/usr/bin/env bash

IMAGE_VERSION=0.3.2-cuda12.1.0-cudnn8-torch2.1.2-cluster-all-1.0.3
IMAGE_NAME=10.108.0.3:15080/zhaobing/vllm
DOCKER_NAME=qwen-72b
## 挂载路径，是模型路径的上一级
MNT_PATH=/mnt/geogpt-gpfs/ali/models/Qwen/Qwen-geo
## 模型当前路径名称
MODEL_NAME=geogpt-72b-0326
## 使用的显卡
DEVICES='"device=0,1,2,3,4,5,6,7"'
POD_NAME=**
SERVICE_NAME=**
GROUP_SIZE=1
DEPLOY_MODE="docker"

start() {
    # docker start command
    docker run -d --name ${DOCKER_NAME} \
        --log-opt max-size=30m \
        --log-opt max-file=3 \
        --gpus=${DEVICES} \
        --shm-size=16g \
        --network host \
        -e NVIDIA_REQUIRE_CUDA="cuda>=11.1" \
        -e POD_NAME=${POD_NAME} \
        -e SERVICE_NAME=${SERVICE_NAME} \
        -e GROUP_SIZE=${GROUP_SIZE} \
        -e DEPLOY_MODE=${DEPLOY_MODE} \
        -e LANG="C.UTF-8" \
        -e LC_ALL="C.UTF-8" \
        -e GPU_USAGE=0.95 \
        -e GROUP_SIZE=${GROUP_SIZE} \
        -e MODEL_PATH=/workspace/models/${MODEL_NAME} \
        -e MODEL_TYPE="version2" \
        -v ${MNT_PATH}:/workspace/models \
        ${IMAGE_NAME}:${IMAGE_VERSION}
}
$1