root@ip-10-37-53-67:/home/ubuntu# cat /opt/data/scripts/vllm.sh
#!/usr/bin/env bash

IMAGE_VERSION=0.2.2-cuda11.8.0-cudnn8-torch2.0.1
IMAGE_NAME=vllm
CONTAINER_NAME=vllm72b
MODEL_DIR=/opt/data/models/geo_72B_sft_ckpt_mp8_pp4_hf-2
DEVICES='"device=0,1,2,3,4,5,6,7"'

start() {
    # docker start command
    docker run -d --name ${CONTAINER_NAME} \
        --log-opt max-size=30m \
        --log-opt max-file=3 \
        --gpus=${DEVICES} \
        --shm-size=16g \
        --network host \
        -e LANG="C.UTF-8" \
        -e LC_ALL="C.UTF-8" \
        -e MODEL_PATH=${MODEL_DIR} \
        -e MODEL_TYPE="llm" \
        -v ${MODEL_DIR}:${MODEL_DIR} \
        ${IMAGE_NAME}:${IMAGE_VERSION}
}

$1
