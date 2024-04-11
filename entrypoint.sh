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

gpu_usage=0.92
max_num_seqs=128
port=8000
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
python3 -m vllm.entrypoints.openai.api_server --port ${port} --host 0.0.0.0 --gpu-memory-utilization ${gpu_usage} --tensor-parallel-size=${TP_SIZE} --served-model-name ${MODEL_TYPE} --model ${MODEL} --trust-remote-code --max-num-seqs=${max_num_seqs} --max-log-len 10
