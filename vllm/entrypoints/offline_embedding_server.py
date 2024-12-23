"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import json
import ssl
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm import LLM
from transformers import AutoTokenizer

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, random_uuid
import time
import struct

logger = init_logger("vllm.entrypoints.api_server")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
task = 'Given a web search query, retrieve relevant passages that answer the query'
max_length = 4096


@app.post("/embeddings")
async def create_embeddings(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompts = request_dict.pop("prompts")
    prompts = [get_detailed_instruct(task, it) for it in prompts]
    llm_engine_prompt_encode_id_dict = [{"prompt_token_ids": tokenizer(prompt, max_length=max_length - 1, padding=True, truncation=True, return_tensors="pt")["input_ids"][0]} for prompt in prompts]
    outputs = model.encode(llm_engine_prompt_encode_id_dict)
    # res = [output.outputs.embedding for output in outputs]
    header = struct.pack('II', len(outputs), len(outputs[0].outputs.embedding) if outputs[0] else 0)

    body = [itm for output in outputs for itm in output.outputs.embedding]
    # print(res)
    return Response(content=header + struct.pack(f'{len(body)}f', *body))


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default='/nas/czh/sfr/SFR-Embedding-Mistral')
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="info")
    
    args = parser.parse_args()
    model = LLM(model=args.model, disable_sliding_window=True, task="embedding", tensor_parallel_size=args.tensor_parallel_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # add_eos_token is set to true in tokenizer_config.json
    # tokenizer.add_eos_token = True

    app.root_path = args.root_path

    logger.info("Available routes are:")
    for route in app.routes:
        if not hasattr(route, 'methods'):
            continue
        methods = ', '.join(route.methods)
        logger.info("Route: %s, Methods: %s", route.path, methods)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
