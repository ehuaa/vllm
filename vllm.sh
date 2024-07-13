#!/usr/bin/env bash

IMAGE_VERSION=latest
IMAGE_NAME=vllm/vllm-v0.5.1-qwen2
CONTAINER_NAME=vllm72b-zj-deploy
MODEL_DIR=/nas/dch/models/Qwen2-7B-Instruct
DEVICES='"device=1,2,3,4"'

start() {
    # docker start command
    # -e ENABLE_PREFIX_CACHING=1 \
    # -e ENABLE_CHUNKED_PREFILL=1 \
    # seems like flashinfer does not support prefix caching or chunked prefilling
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
        -e KV_CACHE_DTYPE="auto" \
        -e GPU_USAGE=0.9 \
        -e VLLM_ATTENTION_BACKEND="FLASHINFER" \
        -v ${MODEL_DIR}:${MODEL_DIR} \
        ${IMAGE_NAME}:${IMAGE_VERSION}
}

$1
