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
max_num_seqs=256
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

if [ -n "$DISABLE_CUSTOM_ALL_REDUCE" ]; then
    DISABLE_CUSTOM_ALL_REDUCE_ARG="--disable-custom-all-reduce"
else
    DISABLE_CUSTOM_ALL_REDUCE_ARG=""
fi

if [ -n "$DISABLE_SLIDING_WINDOW" ]; then
    DISABLE_SLIDING_WINDOW_ARG="--disable-sliding-window"
else
    DISABLE_SLIDING_WINDOW_ARG=""
fi

if [ -n "$ENFORCE_EAGER" ]; then
    ENFORCE_EAGER_ARG="--enforce-eager"
else
    ENFORCE_EAGER_ARG=""
fi

if [ -n "$DISABLE_LOG_REQUESTS" ]; then
    DISABLE_LOG_ARG="--disable-log-requests"
else
    DISABLE_LOG_ARG=""
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

if [ -n "$EMBEDDING_OFFLINE" ]; then
    python3 -m vllm.entrypoints.offline_embedding_server --port ${port} --host 0.0.0.0 --model ${MODEL} $ENFORCE_EAGER_ARG $DISABLE_LOG_ARG $DISABLE_SLIDING_WINDOW_ARG
elif [ -n "$EMBEDDING_ONLINE" ]; then
    python3 -m vllm.entrypoints.openai.api_server --port ${port} --host 0.0.0.0 --model ${MODEL} $ENFORCE_EAGER_ARG $DISABLE_LOG_ARG $DISABLE_SLIDING_WINDOW_ARG
else
    python3 -m vllm.entrypoints.openai.api_server --port ${port} --host 0.0.0.0 \
    --gpu-memory-utilization ${gpu_usage} --tensor-parallel-size=${TP_SIZE} --served-model-name ${MODEL_TYPE} \
    --model ${MODEL} --trust-remote-code --max-num-seqs=${max_num_seqs} --kv-cache-dtype ${kv_cache_dtype} $CHUNKED_PREFILL_ARG $PREFIX_CACHING_ARG $DISABLE_CUSTOM_ALL_REDUCE_ARG $DISABLE_SLIDING_WINDOW_ARG $ENFORCE_EAGER_ARG
fi