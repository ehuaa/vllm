import asyncio
import codecs
import json
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
import importlib
import inspect
import multiprocessing
import os
import re
import signal
import socket
import tempfile
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import partial
from http import HTTPStatus
from typing import AsyncIterator, Set

from collections import deque
import itertools

from prometheus_client import make_asgi_app

import uvloop
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import State
from starlette.routing import Mount
from typing_extensions import assert_never

from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest,
                                              EmbeddingResponse, ErrorResponse,
                                              LoadLoraAdapterRequest,
                                              TokenizeRequest,
                                              TokenizeResponse,
                                              UnloadLoraAdapterRequest)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_open_zmq_ipc_path
from vllm.version import __version__ as VLLM_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
task = 'Given a web search query, retrieve relevant passages that answer the query'
max_length = 4096

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()
served_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(10.)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


@asynccontextmanager
async def build_async_engine_client(
        args: Namespace) -> AsyncIterator[EngineClient]:

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)

    async with build_async_engine_client_from_engine_args(
            engine_args, args.disable_frontend_multiprocessing) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Fall back
    # TODO: fill out feature matrix.
    if (MQLLMEngineClient.is_unsupported_config(engine_args)
            or disable_frontend_multiprocessing):
        engine_config = engine_args.create_engine_config()
        uses_ray = getattr(AsyncLLMEngine._get_executor_cls(engine_config),
                           "uses_ray", False)

        build_engine = partial(AsyncLLMEngine.from_engine_args,
                               engine_args=engine_args,
                               engine_config=engine_config,
                               usage_context=UsageContext.OPENAI_API_SERVER)
        if uses_ray:
            # Must run in main thread with ray for its signal handlers to work
            engine_client = build_engine()
        else:
            engine_client = await asyncio.get_running_loop().run_in_executor(
                None, build_engine)

        yield engine_client
        return

    # Otherwise, use the multiprocessing AsyncLLMEngine.
    else:
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            # Make TemporaryDirectory for prometheus multiprocessing
            # Note: global TemporaryDirectory will be automatically
            #   cleaned up upon exit.
            global prometheus_multiproc_dir
            prometheus_multiproc_dir = tempfile.TemporaryDirectory()
            os.environ[
                "PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        else:
            logger.warning(
                "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                "This directory must be wiped between vLLM runs or "
                "you will find inaccurate metrics. Unset the variable "
                "and vLLM will properly handle cleanup.")

        # Select random path for IPC.
        ipc_path = get_open_zmq_ipc_path()
        logger.info("Multiprocessing frontend to use %s for IPC Path.",
                    ipc_path)

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        context = multiprocessing.get_context("spawn")

        engine_process = context.Process(target=run_mp_engine,
                                         args=(engine_args,
                                               UsageContext.OPENAI_API_SERVER,
                                               ipc_path))
        engine_process.start()
        logger.info("Started engine process with PID %d", engine_process.pid)

        # Build RPCClient, which conforms to EngineClient Protocol.
        # NOTE: Actually, this is not true yet. We still need to support
        # embedding models via RPC (see TODO above)
        engine_config = engine_args.create_engine_config()
        mp_engine_client = MQLLMEngineClient(ipc_path, engine_config)

        try:
            while True:
                try:
                    await mp_engine_client.setup()
                    break
                except TimeoutError:
                    if not engine_process.is_alive():
                        raise RuntimeError(
                            "Engine process failed to start") from None

            yield mp_engine_client  # type: ignore[misc]
        finally:
            # Ensure rpc server process was terminated
            engine_process.terminate()

            # Close all open connections to the backend
            mp_engine_client.close()

            # Wait for engine process to join
            engine_process.join(4)
            if engine_process.exitcode is None:
                # Kill if taking longer than 5 seconds to stop
                engine_process.kill()

            # Lazy import for prometheus multiprocessing.
            # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
            # before prometheus_client is imported.
            # See https://prometheus.github.io/client_python/multiprocess/
            from prometheus_client import multiprocess
            multiprocess.mark_process_dead(engine_process.pid)


router = APIRouter()


def mount_metrics(app: FastAPI):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import (CollectorRegistry, make_asgi_app,
                                   multiprocess)

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.info("vLLM to use %s as PROMETHEUS_MULTIPROC_DIR",
                    prometheus_multiproc_dir_path)
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    else:
        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def chat(request: Request) -> OpenAIServingChat:
    return request.app.state.openai_serving_chat


def completion(request: Request) -> OpenAIServingCompletion:
    return request.app.state.openai_serving_completion


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def embedding(request: Request) -> OpenAIServingEmbedding:
    return request.app.state.openai_serving_embedding


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client

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
        
def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message,
                             type=err_type,
                             code=status_code.value)    


async def get_gen_prompt_ids_with_history(messages, prompt, system_message, 
                                      max_window_size=3000, 
                                      max_content_round=20) -> str:
    if args.served_model_name[0].startswith("Qwen"):
        sep = ("<|im_end|>", "<|im_end|>")
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        has_seperate = "\n"
        tmp_ret = "<|im_start|>system\n{system_message}".format(system_message=system_message) + sep[0] + "\n"
    elif args.served_model_name[0].startswith("Llama3"):
        sep = ("<|eot_id|>", "<|eot_id|>")
        roles = ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n")
        has_seperate = ""
        tmp_ret = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}".format(system_message=system_message) + sep[0]
    elif args.served_model_name[0].startswith("Mixtral"):
        sep = (" [/INST]", "</s>")
        roles = (" [INST] ", " ")
        has_seperate = ""
        tmp_ret = "<s> [INST] {system_message}\n\n".format(system_message=system_message)

    token_ids = tokenizer(tmp_ret).input_ids
    token_num = len(token_ids)
    content_round = 0
    tmp_ret = ""

    # add chat history to input_ids
    history_token_ids = deque()
    for msg_user, msg_assistant in reversed(messages):
        if not args.served_model_name[0].startswith("Mixtral"):
            if msg_user:
                tmp_ret += roles[0] + msg_user + sep[0] + has_seperate
            else:
                tmp_ret += roles[0]

            if msg_assistant:
                tmp_ret += roles[1] + msg_assistant + sep[1] + has_seperate
            else:
                tmp_ret += roles[1]
        else:
            tmp_ret += msg_user + sep[0] + has_seperate
            tmp_ret += roles[1] + msg_assistant + sep[1] + roles[0]

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
    if args.served_model_name[0].startswith("Mixtral"):
        tmp_ret += prompt + sep[0]
    else:
        tmp_ret += roles[0] + prompt + sep[0] + has_seperate
        tmp_ret += roles[1]

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
        return input_ids, create_error_response(
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
        return prompt_ids, create_error_response(
            message=
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {params['max_tokens'] + token_num} tokens "
            f"({token_num} in the messages, "
            f"{params['max_tokens']} in the completion). "
            f"Please reduce the length of the messages or completion."
        )
    else:
        return prompt_ids, None
    
    
@router.post("/llm/generate")
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
    stop_words = request_dict.pop("stopWords", [])

    # generate prompt
    prompt_ids = await get_gen_prompt_ids_with_history(chat_history, prompt, system_message, max_window_size, max_content_round)
    prompt_ids, error_check_ret = await check_qwen_length(prompt_ids, sampling_params, max_length)

    if error_check_ret is not None:
        logger.error("requestId: " + str(request_id) + "\n" + error_check_ret.message)
        return error_check_ret
    
    stop = []
    if tokenizer.added_tokens_encoder is not None:
        stop.extend(list(tokenizer.added_tokens_encoder.keys()))
    else:
        stop.append(tokenizer.eos_token)
        
    if stop_words:
        for singleStopWord in stop_words:
            stop.append(singleStopWord)
    sampling_params["stop"] = stop
    
    # best_of param should be set to 1 when not using beam search, in case of passing in an outlier best_of value 
    # which will stuck the main process for very long time.
    # Here we set best_of to None forcely when not using beam search.
    if not sampling_params.get("use_beam_search", False):
        sampling_params["best_of"] = None
     
    sampling_params = SamplingParams(**sampling_params)
    results_generator = engine_client.generate({"prompt_token_ids": prompt_ids}, sampling_params, request_id)

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
            await engine_client.abort(request_id)
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

@router.get("/ready")
async def ready(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)

@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.post("/tokenize")
async def tokenize(request: TokenizeRequest, raw_request: Request):
    generator = await tokenization(raw_request).create_tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/detokenize")
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    generator = await tokenization(raw_request).create_detokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    models = await completion(raw_request).show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):

    generator = await chat(raw_request).create_chat_completion(
        request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await completion(raw_request).create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await embedding(raw_request).create_embedding(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!")

    @router.post("/start_profile")
    async def start_profile(raw_request: Request):
        logger.info("Starting profiler...")
        await engine_client(raw_request).start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile(raw_request: Request):
        logger.info("Stopping profiler...")
        await engine_client(raw_request).stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


if envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
    logger.warning(
        "Lora dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!")

    @router.post("/v1/load_lora_adapter")
    async def load_lora_adapter(request: LoadLoraAdapterRequest,
                                raw_request: Request):
        response = await chat(raw_request).load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        response = await completion(raw_request).load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)

    @router.post("/v1/unload_lora_adapter")
    async def unload_lora_adapter(request: UnloadLoraAdapterRequest,
                                  raw_request: Request):
        response = await chat(raw_request).unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        response = await completion(raw_request).unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(),
                                status_code=response.code)

        return Response(status_code=200, content=response)


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(openapi_url=None,
                      docs_url=None,
                      redoc_url=None,
                      lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        chat = app.state.openai_serving_chat
        err = chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

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

    return app


def init_app_state(
    engine_client: EngineClient,
    model_config: ModelConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model)
        for name in served_model_names
    ]
    
    global tokenizer, max_model_len
    max_model_len = model_config.max_model_len
    tokenizer = get_tokenizer(
        model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code)
    load_chat_template(args, tokenizer)
    
    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats

    state.openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        base_model_paths,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser)
    state.openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )
    state.openai_serving_embedding = OpenAIServingEmbedding(
        engine_client,
        model_config,
        base_model_paths,
        request_logger=request_logger,
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valide_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
        and args.tool_call_parser not in valide_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valide_tool_parses)} }})")

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", args.port))

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    global engine_client
    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        init_app_state(engine_client, model_config, app.state, args)

        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            fd=sock.fileno(),
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
