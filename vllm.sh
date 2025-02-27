#!/usr/bin/env bash

IMAGE_VERSION=latest
IMAGE_NAME=vllm/vllm-openai-fix
CONTAINER_NAME=vllm72b-zj-offline-embedding
# MODEL_DIR=/data/xq/qwen2-5-72b-dpo-1101
MODEL_DIR=/data/czh/SFR-Embedding-Mistral
CODE_DIR=/data/czh/vllm
DEVICES='"device=0,1,2,3,4,5,6,7"'

start() {
    # docker start command
    # -e ENABLE_PREFIX_CACHING=1 \
    # -e ENABLE_CHUNKED_PREFILL=1 \
    # -e VLLM_ATTENTION_BACKEND="FLASHINFER" \
    # -e DISABLE_SLIDING_WINDOW=1 \
    # -e ENFORCE_EAGER=1 \
    # seems like flashinfer does not support prefix caching or chunked prefilling
    echo "start run docker..."
    # docker run -d --name ${CONTAINER_NAME} \
    #     --log-opt max-size=30m \
    #     --log-opt max-file=3 \
    #     --gpus=${DEVICES} \
    #     --shm-size=16g \
    #     --network host \
    #     -e LANG="C.UTF-8" \
    #     -e LC_ALL="C.UTF-8" \
    #     -e MODEL_PATH=${MODEL_DIR} \
    #     -e MODEL_TYPE="Qwen2" \
    #     -e PORT=8809 \
    #     -e KV_CACHE_DTYPE="auto" \
    #     -e GPU_USAGE=0.9 \
    #     -e ENABLE_CHUNKED_PREFILL=1 \
    #     -v ${MODEL_DIR}:${MODEL_DIR} \
    #     -v ${CODE_DIR}:${CODE_DIR} \
    #     ${IMAGE_NAME}:${IMAGE_VERSION}
    docker run -d --name ${CONTAINER_NAME} \
        --log-opt max-size=30m \
        --log-opt max-file=3 \
        --gpus=${DEVICES} \
        --shm-size=16g \
        --network host \
        -e MODEL_PATH=${MODEL_DIR} \
        -e EMBEDDING_OFFLINE=1 \
        -e PORT=28192 \
        -v ${MODEL_DIR}:${MODEL_DIR} \
        -v ${CODE_DIR}:${CODE_DIR} \
        ${IMAGE_NAME}:${IMAGE_VERSION}
}

$1
