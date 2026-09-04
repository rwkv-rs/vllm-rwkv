# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.entrypoints.openai.chat_completion.batch_serving import OpenAIServingChatBatch
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    validate_json_request,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.orca_metrics import metrics_header
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()
ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def batch_chat(request: Request) -> OpenAIServingChatBatch | None:
    return request.app.state.openai_serving_chat_batch


async def _rwkv_state_action(
    raw_request: Request,
    action: str,
    state_ref: str = "",
    target_ref: str = "",
):
    handler = chat(raw_request)
    if handler is None:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            detail="Chat serving is unavailable",
        )
    try:
        return await handler.rwkv_state_cache_action(action, state_ref, target_ref)
    except Exception as error:
        detail = str(error)
        if isinstance(error, KeyError) or "Unknown RWKV State ref" in detail:
            status_code = HTTPStatus.NOT_FOUND
        elif (
            isinstance(error, (ValueError, MemoryError))
            or "RWKV State ref is in use" in detail
            or "RWKV State ref already exists" in detail
        ):
            status_code = HTTPStatus.CONFLICT
        elif isinstance(error, NotImplementedError):
            status_code = HTTPStatus.NOT_IMPLEMENTED
        else:
            status_code = HTTPStatus.SERVICE_UNAVAILABLE
        raise HTTPException(status_code=status_code, detail=detail) from error


@router.get("/v1/rwkv/state/capabilities")
async def rwkv_state_capabilities(raw_request: Request):
    return await _rwkv_state_action(raw_request, "capabilities")


@router.get("/v1/rwkv/state/{state_ref}")
async def inspect_rwkv_state(state_ref: str, raw_request: Request):
    return await _rwkv_state_action(raw_request, "inspect", state_ref)


@router.post("/v1/rwkv/state/{state_ref}/clone")
async def clone_rwkv_state(
    state_ref: str,
    payload: dict[str, str],
    raw_request: Request,
):
    return await _rwkv_state_action(
        raw_request,
        "clone",
        state_ref,
        str(payload.get("target_ref") or ""),
    )


@router.delete("/v1/rwkv/state/{state_ref}")
async def drop_rwkv_state(state_ref: str, raw_request: Request):
    return await _rwkv_state_action(raw_request, "drop", state_ref)


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    generator = await handler.create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions/batch",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
        HTTPStatus.NOT_IMPLEMENTED.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_batch_chat_completion(
    request: BatchChatCompletionRequest, raw_request: Request
):
    handler = batch_chat(raw_request)
    if handler is None:
        raise NotImplementedError("The model does not support Chat Completions API")

    result = await handler.create_batch_chat_completion(request, raw_request)

    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)

    return JSONResponse(content=result.model_dump())


def attach_router(app: FastAPI):
    app.include_router(router)
