"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests
import uuid


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)

with open('/mnt/geogpt-gpfs/zhijiang/home/czh/vllm/vllm_example_zj.json') as f:
        data = json.load(f)

with open('/mnt/geogpt-gpfs/zhijiang/home/czh/vllm/lqa_v2.0.json') as f:
        data1 = json.load(f)
print(data1)

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
    data["data"]["input"] = question
    data["data"]["requestId"] = str(uuid.uuid4().hex)
    response = requests.post(api_url, headers=headers, json=data, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["data"]["output"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["data"]["output"]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="10.244.127.79")
    parser.add_argument("--port", type=int, default=8000)
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
                for i, line in enumerate(h):
                    num_printed_lines += 1
                    print(f"Beam candidate {i}: {line!r}", flush=True)
        else:
            output = get_response(response)
            # for i, line in enumerate(output):
            #     pass
            #     # print(f"Beam candidate {i}: {line!r}", flush=True)
            # print(output)
            question["answer"] = output
            question["answer_length"] = len(output)
                                        
    
    with open('vllm_ali_res.json', 'w') as file:
        json.dump(data1, file)
