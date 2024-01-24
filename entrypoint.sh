TP_SIZE=$(python3 gpu_count.py)
echo "TP_SIZE is $TP_SIZE"
#cd /workspace/vllm
wssh --port=8003 --fbidhttp=False >webssh.log 2>&1 &
MODEL=""
if [ "$MODEL_DIR" != "" ];then
    MODEL=$MODEL_DIR
elif [ "$MODEL_PATH" != "" ];then
    MODEL=$MODEL_PATH
else
    echo "MODEL_PATH or MODEL_DIR is need, current is $MODEL_PATH and $MODEL_DIR"
fi
echo "MODEL config is $MODEL"

gpu_usage=0.9
if [ -n "$GPU_USAGE" ]; then
   gpu_usage=$GPU_USAGE
fi
echo "GPU_USAGE is $gpu_usage"
#while true; do sleep 1s; done;
python3 -m vllm.entrypoints.api_server --port 8000 --host 0.0.0.0 --gpu-memory-utilization ${gpu_usage} -tp=${TP_SIZE} --model_type ${MODEL_TYPE} --model ${MODEL} --trust-remote-code
