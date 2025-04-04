# SPDX-License-Identifier: Apache-2.0
"""Example Python client for `vllm.entrypoints.api_server`
NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend `vllm serve` and the OpenAI client API.
"""

import argparse
import json
from collections.abc import Iterable

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)

with open('/root/vllm_test/czh/vllm/vllm_example_zj1.json') as f:
        data = json.load(f)

with open('/root/vllm_test/czh/vllm/lqa_v2.0.json') as f:
        data1 = json.load(f)
# print(data1)

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
    # data["data"]["input"] = question
    data = {
        "input": "who are you?",
        "serviceParams": {
            "promptTemplateName": "geogpt",
            "stream":False,
            "maxOutputLength": 3000
        },
        "history": [],
        "modelParams": {
            "temperature": 0.6,
            "presence_penalty": 2.0,
            "top_p": 0.8,
        }
    }
    response = requests.post(api_url, headers=headers, json=data, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[list[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            # print(chunk)
            output = json.loads(chunk.decode("utf-8"))            
            yield output


def get_response(response: requests.Response) -> list[str]:
    data = json.loads(response.content)
    output = data["data"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="10.200.99.220")
    parser.add_argument("--port", type=int, default=30106)
    # parser.add_argument("--host", type=str, default="10.244.127.79")
    # parser.add_argument("--port", type=int, default=8000)
    # parser.add_argument("--n", type=int, default=4)
    # parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    # prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/llm/generate"
    # n = args.n
    stream = args.stream

    # print(f"Prompt: {prompt!r}\n", flush=True)
    for question in data1:
        response = post_http_request(api_url, question["question"])

        if stream:
            num_printed_lines = 0
            for h in get_streaming_response(response):
                clear_line(num_printed_lines)
                num_printed_lines = 0
                num_printed_lines += 1
                print(f"{h.get('output', '')!r}", flush=True)
        else:
            output = get_response(response)
            # for i, line in enumerate(output):
            #     pass
            #     # print(f"Beam candidate {i}: {line!r}", flush=True)
            # print(output)
            print(output)
            question["answer"] = output
            question["answer_length"] = len(output)
                                        
    
    with open('vllm_zj_res.json', 'w') as file:
        json.dump(data1, file)
