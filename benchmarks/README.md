# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Zhejianglab benchmark serving with benchmark_serving_zj.py, while the original benchmark_serving.py has already had FTL, we will be compatiable with it later.(Update 2024/3/11)