# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
import asyncio
import codecs
import json
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
import os
import importlib
import inspect
import re
from typing import Optional, Set

from collections import deque
import itertools

from prometheus_client import make_asgi_app
import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid
from vllm.entrypoints.openai.cli_args import make_arg_parser
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest, ErrorResponse,
                                              TokenizeRequest,
                                              TokenizeResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.version import __version__ as VLLM_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_embedding: OpenAIServingEmbedding

logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()

served_model = None
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


# app = fastapi.FastAPI(lifespan=lifespan)
app = fastapi.FastAPI()
engine = None
response_role = None


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)


def load_chat_template(args, tokenizer):
    if args.chat_template is not None:
        try:
            with open(args.chat_template, "r") as f:
                chat_template = f.read()
        except OSError:
            # If opening a file fails, set chat template to be args to
            # ensure we decode so our escape are interpreted correctly
            chat_template = codecs.decode(args.chat_template, "unicode_escape")

        tokenizer.chat_template = chat_template
        logger.info(
            f"Using supplied chat template:\n{tokenizer.chat_template}")
    elif tokenizer.chat_template is not None:
        logger.info(f"Using default chat template:\n{tokenizer.chat_template}")
    else:
        logger.warning("No chat template provided. Chat API will not work.")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


async def get_gen_prompt_ids_with_history(messages, prompt, system_message, 
                                      max_window_size=3000, 
                                      max_content_round=20) -> str:
    sep = "<|im_end|>"
    roles = ("<|im_start|>user", "<|im_start|>assistant")
    
    tmp_ret = "<|im_start|>system\n{system_message}".format(system_message=system_message) + sep + "\n"
    token_ids = tokenizer(tmp_ret).input_ids
    token_num = len(token_ids)
    content_round = 0
    tmp_ret = ""
    
    # add chat history to input_ids
    history_token_ids = deque()
    for msg_user, msg_assistant in reversed(messages):
        if msg_user:
            tmp_ret += roles[0] + "\n" + msg_user + sep + "\n"
        else:
            tmp_ret += roles[0] + "\n"
            
        if msg_assistant:
            tmp_ret += roles[1] + "\n" + msg_assistant + sep + "\n"
        else:
            tmp_ret += roles[1] + "\n"
        
        tmp_token_ids = tokenizer(tmp_ret).input_ids
        content_round += 1
        token_num += len(tmp_token_ids)
        if token_num > max_window_size or content_round > max_content_round:
            break
        else:
            history_token_ids.appendleft(tmp_token_ids)
            tmp_ret = ""
    
    history_token_ids.appendleft(token_ids)
    # add prompt to token_ids
    tmp_ret = ""
    tmp_ret += roles[0] + "\n" + prompt + sep + "\n"
    tmp_ret += roles[1] + "\n"
    history_token_ids.append(tokenizer(tmp_ret).input_ids)
    
    token_ids = list(itertools.chain(*list(history_token_ids)))
    
    return token_ids


async def check_length(
    request: Union[ChatCompletionRequest, CompletionRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    input_ids = prompt_ids if prompt_ids is not None else tokenizer(
        prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        return input_ids, openai_serving_chat.create_error_response(
            message=
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion."
        )
    else:
        return input_ids, None


async def check_qwen_length(
    prompt_ids: Optional[List[int]] = None,
    params: Dict = None,
    max_length: int = 8192
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (prompt_ids is not None and params is not None), "prompt_ids and params should be provided."
    token_num = len(prompt_ids)
    
    if "max_tokens" not in params:
        params["max_tokens"] = max_model_len - token_num
    params["max_tokens"] = min(max_length, params["max_tokens"])
    if params["max_tokens"] <= 0 or token_num + params["max_tokens"] > max_model_len:
        return prompt_ids, openai_serving_chat.create_error_response(
            message=
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {params['max_tokens'] + token_num} tokens "
            f"({token_num} in the messages, "
            f"{params['max_tokens']} in the completion). "
            f"Please reduce the length of the messages or completion."
        )
    else:
        return prompt_ids, None


@app.get("/ready")
async def ready() -> JSONResponse:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return JSONResponse(content={"code":200, "message":"success", "data":None})


@app.get("/health")
async def health() -> JSONResponse:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return JSONResponse(content={"code":200, "message":"success", "data":None})


@app.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    generator = await openai_serving_completion.create_tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, TokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@app.post("/detokenize")
async def detokenize(request: DetokenizeRequest):
    generator = await openai_serving_completion.create_detokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, DetokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.post("/llm/generate")
async def llm_generate(request: Request) -> Response:
    """
    Specify Use for Qwen style chat LLM for 14B or 72B
    Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    request_dict = request_dict.pop("data")
    logger.info(
            f"llm_generate receive request dict:\n{request_dict}")
    # get data from request_dict
    stream = request_dict.pop("stream", False)
    chat_history = request_dict.pop("history", [])
    if chat_history is None:
        chat_history = []
    system_message = request_dict.pop("system", "")
    prompt = request_dict.pop("input", "")
    max_window_size = request_dict.pop("maxWindowSize", 3000)
    max_content_round = request_dict.pop("maxContentRound", 20)
    max_length = request_dict.pop("maxLength", 8192)
    sampling_params = request_dict.pop("params")
    request_id = request_dict.pop("requestId", random_uuid())

    # generate prompt
    prompt_ids = await get_gen_prompt_ids_with_history(chat_history, prompt, system_message, max_window_size, max_content_round)
    prompt_ids, error_check_ret = await check_qwen_length(prompt_ids, sampling_params, max_length)

    if error_check_ret is not None:
        logger.error("requestId: " + str(request_id) + "\n" + error_check_ret.message)
        return error_check_ret
    
    sampling_params["stop"] = ["<|im_end|>"]
    
    # best_of param should be set to 1 when not using beam search, in case of passing in an outlier best_of value 
    # which will stuck the main process for very long time.
    # Here we set best_of to None forcely when not using beam search.
    if not sampling_params.get("use_beam_search", False):
        sampling_params["best_of"] = None
     
    sampling_params = SamplingParams(**sampling_params)
    results_generator = engine.generate({"prompt_token_ids": prompt_ids}, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            # prompt = request_output.prompt
            text_outputs = [
                output.text for output in request_output.outputs
            ]
            ret = {"output": text_outputs[0]}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    # prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"requestId":request_id,
           "code":"200" ,
           "message":"success",
           "data": {"output": text_outputs[0]}}
    return JSONResponse(ret)


@app.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await openai_serving_embedding.create_embedding(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    response_role = args.response_role

    engine_args = AsyncEngineArgs.from_cli_args(args)

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)

    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())
    max_model_len = model_config.max_model_len

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(
        model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code)
    load_chat_template(args, tokenizer)
    
    openai_serving_chat = OpenAIServingChat(engine, model_config,
                                            served_model_names,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, model_config, served_model_names, args.lora_modules)
    openai_serving_embedding = OpenAIServingEmbedding(engine, model_config,
                                                      served_model_names)
    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
