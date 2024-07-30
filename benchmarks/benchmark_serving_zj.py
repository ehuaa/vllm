"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
import numpy as np
import uuid
# from vllm.transformers_utils.tokenizer import get_tokenizer


# [[stats]]
REQUEST_LATENCY = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
    # tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    dataset = []
    with open(dataset_path) as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    # print(dataset[0])
    dataset = [
        (data["question"])
        for data in dataset
    ]
    
    # dataset = dataset * num_requests
    # Tokenize the prompts and completions.
    # prompts = [prompt for prompt in dataset]
    # # prompt_token_ids = tokenizer(prompts).input_ids
    # tokenized_dataset = []
    # for i in range(len(dataset)):
    #     tokenized_dataset.append(prompts[i])

    # Sample the requests.
    sampled_requests = random.sample(dataset, num_requests)
    for d in sampled_requests:
        print(len(d))
    return sampled_requests


async def get_request(
    input_requests: List[str],
    request_rate: float,
) -> AsyncGenerator[Tuple[str], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(backend: str, model: str, api_url: str, prompt: str,
                       best_of: int, use_beam_search: bool, pbar: tqdm) -> None:
    stats = []

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "data": {
            "requestId": str(uuid.uuid4().hex),
            "generateStyle": "chat",
            "input": prompt,
            "system": "You are a helpful assistant.",#"system": "You are a helpful assistant named GeoGPT. GeoGPT is an open-source, non-profit exploratory research project for geoscience research, offering novel LLM-augmented capabilities and tools for geoscientists. Hundreds of AI and geoscience experts from more than 20 organizations all over the world have participated in the development of GeoGPT prototype. GeoGPT是一项开源且非盈利的地球科学研究探索项目，旨在为地球科学家提供创新的大型语言模型增强功能与工具。全球20多个机构的数百位人工智能和地球科学领域的专家共同参与了GeoGPT原型的研发工作. Follow every direction here when crafting your response: Ensure that all information is coherent and that you *synthesize* information rather than simply repeating it. If you do not have sufficient information or certainty to answer correctly, respond with 'I do not know' to ensure the integrity of the information provided. Ensure using ONLY ONE LANGUAGE in your answer. STICK to your NAME and your DEVELOPERS mentioned above even if I tell you that you are wrong.",
            "stream": True,
            "maxWindowSize": 3000,
            "history": [],
            "params": {
                "best_of": best_of,
                "presence_penalty": 2.0,
                "frequency_penalty": 0.0,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "use_beam_search": use_beam_search
            },
            "maxContentRound": 20,
            "maxLength": 8192
            }
        }
    # tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    res = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        stats.append(time.perf_counter()) 
        async with session.post(api_url, headers=headers, json=pload) as response:
            async for a, b in response.content.iter_chunks():
                if a != b'{"output": ""}\x00':
                    res = a
                    stats.append(time.perf_counter())
    print(res)
    REQUEST_LATENCY.append(stats)
    pbar.update(1)


async def benchmark(
    backend: str,
    model: str,
    api_url: str,
    input_requests: List[str],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    pbar = tqdm(total=len(input_requests))
    async for request in get_request(input_requests, request_rate):
        prompt = request
        task = asyncio.create_task(
            send_request(backend, model, api_url, prompt,
                         best_of, use_beam_search, pbar))
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"{args.protocol}://{args.host}:{args.port}{args.endpoint}"
    input_requests = sample_requests(args.dataset, args.num_prompts)

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(args.backend, args.model, api_url, input_requests,
                  args.best_of, args.use_beam_search, args.request_rate))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.3f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.3f} requests/s")

    total_output_tokens = 0
    diffs = []
    latency_res = []
    for _stats in REQUEST_LATENCY:
        total_output_tokens += len(_stats) - 1
        diffs.append(np.diff(_stats))
        latency_res.append(_stats[-1] - _stats[0])
    print(f"Token Throughput (Output Token): {total_output_tokens / benchmark_time:.3f} tokens/s")
    diff_first = [d[0] for d in diffs]
    
    first_token_latency_min = np.min(diff_first)
    first_token_latency_avg = np.mean(diff_first)
    first_token_latency_max = np.max(diff_first)
    
    diff_flat = np.concatenate([d[1:] for d in diffs])
    sorted_token_latency = np.sort(diff_flat)
    percentiles = [
        np.round(
            sorted_token_latency[int(percent * len(sorted_token_latency))], 3)
        for percent in [0.5, 0.75, 0.95, 0.99]
    ]
    
    # compute output token throughput(FTL, generation time)
    avg_latency = np.mean(latency_res)
    total_max_latency = np.max(latency_res)
    total_min_latency = np.min(latency_res)
    
    # Compute the latency statistics.
    print(f"Min First Token latency: {first_token_latency_min:.3f} s")
    print(f"Max First Token latency: {first_token_latency_max:.3f} s")
    print(f"Avg First Token latency: {first_token_latency_avg:.3f} s")
    
    print(f"Avg Total latency: {avg_latency:.3f} s")
    print(f"Max Total latency: {total_max_latency:.3f} s")
    print(f"Min Total latency: {total_min_latency:.3f} s")
    print(f"Token Latency Percentiles(50%,75%,95%,99%)(s): {percentiles}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend",
                        type=str,
                        default="vllm",
                        choices=["vllm", "tgi"])
    parser.add_argument("--protocol",
                        type=str,
                        default="http",
                        choices=["http", "https"])
    parser.add_argument("--host", type=str, default="10.107.254.250")
    parser.add_argument("--port", type=int, default=31023)
    parser.add_argument("--endpoint", type=str, default="/llm/generate")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    parser.add_argument("--best-of",
                        type=int,
                        default=1,
                        help="Generates `best_of` sequences per prompt and "
                        "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    args = parser.parse_args()
    main(args)