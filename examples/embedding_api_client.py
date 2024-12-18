"""Example Python client for vllm.entrypoints.api_server
NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend vllm.entrypoints.openai.api_server
and the OpenAI client API
"""

import argparse
import json
from typing import Iterable, List

import struct
import requests
import time


def post_http_request(api_url: str, question: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    # pload = {
    #     "prompt": prompt,
    #     "n": n,
    #     "use_beam_search": True,
    #     "temperature": 0.0,
    #     "max_tokens": 16,
    #     "stream": stream,
    # }
    req_data = {"prompts": question}
    response = requests.post(api_url, headers=headers, json=req_data, stream=True)
    return response


def get_response(response: requests.Response) -> List[str]:
    packed_data = response.content
    rows, cols = struct.unpack('II', packed_data[:8])
    
    # unpack data
    flat_data = struct.unpack(f'{rows*cols}f', packed_data[8:])
    
    # reconstruct
    return [list(flat_data[i*cols:(i+1)*cols]) for i in range(rows)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="10.200.48.45")
    parser.add_argument("--port", type=int, default=18192)
    # parser.add_argument("--host", type=str, default="10.244.127.79")
    # parser.add_argument("--port", type=int, default=8000)
    # parser.add_argument("--n", type=int, default=4)
    # parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    # prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/embeddings"
    # n = args.n
    stream = args.stream

    data = []
    with open('/root/vllm_test/czh/vllm/part_1.tsv', 'r', encoding='utf-8') as file:
        for line in file:
            row = line.strip().split('|')[1].strip()
            data.append(row)
    # with open('/nas/czh/sfr/539667.jsonl', 'r', encoding='utf-8') as file:
    #     for line in file:
    #         row = json.loads(line)["text"]
    #         data.append(row)
    # print(f"Prompt: {prompt!r}\n", flush=True)
    t0 = time.perf_counter()

    response = post_http_request(api_url, data)
    output = get_response(response)
    print(time.perf_counter() - t0)
    print(len(output))
    print(output[9][:10])
   
