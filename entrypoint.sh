#!/usr/bin/env bash

echo "run entrypoint..."
TP_SIZE=$(python3 gpu_count.py)
echo "TP_SIZE is $TP_SIZE"
#cd /workspace/vllm
#wssh --port=8003 --fbidhttp=False >webssh.log 2>&1 &
MODEL=""
if [ "$MODEL_DIR" != "" ];then
    MODEL=$MODEL_DIR
elif [ "$MODEL_PATH" != "" ];then
    MODEL=$MODEL_PATH
else
    echo "MODEL_PATH or MODEL_DIR is need, current is $MODEL_PATH and $MODEL_DIR"
fi
echo "MODEL config is $MODEL"

kv_cache_dtype="auto"
gpu_usage=0.92
max_num_seqs=128
port=8000

if [ -n "$KV_CACHE_DTYPE" ]; then
   kv_cache_dtype=$KV_CACHE_DTYPE
fi
echo "kv_cache_dtype is $kv_cache_dtype"

if [ -n "$ENABLE_CHUNKED_PREFILL" ]; then
    CHUNKED_PREFILL_ARG="--enable-chunked-prefill"
else
    CHUNKED_PREFILL_ARG=""
fi

if [ -n "$ENABLE_PREFIX_CACHING" ]; then
    PREFIX_CACHING_ARG="--enable-prefix-caching"
else
    PREFIX_CACHING_ARG=""
fi


if [ -n "$GPU_USAGE" ]; then
   gpu_usage=$GPU_USAGE
fi
echo "GPU_USAGE is $gpu_usage"

if [ -n "$MAX_NUM_SEQS" ]; then
   max_num_seqs=$MAX_NUM_SEQS
fi
echo "MAX_NUM_SEQS is $max_num_seqs"

if [ -n "$PORT" ]; then
   port=$PORT
fi
echo "PORT is $port"
#while true; do sleep 1s; done;
python3 -m vllm.entrypoints.openai.api_server --port ${port} --host 0.0.0.0 \
--gpu-memory-utilization ${gpu_usage} --tensor-parallel-size=${TP_SIZE} --served-model-name ${MODEL_TYPE} \
--model ${MODEL} --trust-remote-code --max-num-seqs=${max_num_seqs} --kv-cache-dtype ${kv_cache_dtype} $CHUNKED_PREFILL_ARG $PREFIX_CACHING_ARG
