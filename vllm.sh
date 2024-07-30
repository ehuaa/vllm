#!/usr/bin/env bash

IMAGE_VERSION=latest
IMAGE_NAME=vllm/vllm-embedding-generation
CONTAINER_NAME=vllm72b-zj-deploy
#MODEL_DIR=/nas/dch/models/Qwen2-7B-Instruct
MODEL_DIR=/nas/czh/sfr/SFR-Embedding-Mistral
CODE_DIR=/nas/czh
DEVICES='"device=0"'

start() {
    # docker start command
    # -e ENABLE_PREFIX_CACHING=1 \
    # -e ENABLE_CHUNKED_PREFILL=1 \
    # -e VLLM_ATTENTION_BACKEND="FLASHINFER" \
    # -e DISABLE_SLIDING_WINDOW=1 \
    # -e ENFORCE_EAGER=1 \
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
        -e MODEL_TYPE="SFR" \
        -e PORT=8810 \
        -e KV_CACHE_DTYPE="auto" \
        -e GPU_USAGE=0.9 \
        -e EMBEDDING_OFFLINE=1 \
        -v ${MODEL_DIR}:${MODEL_DIR} \
        -v ${CODE_DIR}:${CODE_DIR} \
        ${IMAGE_NAME}:${IMAGE_VERSION}
}

$1
