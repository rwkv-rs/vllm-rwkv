#!/usr/bin/env python3
import argparse
import hashlib
import json
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

MODEL_UPSTREAMS = {
    "rwkv7-g1i-1.5b": "http://127.0.0.1:18001",
    "rwkv7-g1i-2.9b": "http://127.0.0.1:18002",
    "rwkv7-g1i-7.2b": "http://127.0.0.1:18003",
    "rwkv7-g1i-13.3b": "http://127.0.0.1:18004",
}
MAX_REQUEST_BYTES = 16 * 1024 * 1024
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error"}},
        status_code=status_code,
    )


def _authorized(request: Request, api_key_sha256: bytes) -> bool:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme != "Bearer" or not token:
        return False
    digest = hashlib.sha256(token.encode()).digest()
    return secrets.compare_digest(digest, api_key_sha256)


def _forward_headers(request: Request) -> dict[str, str]:
    return {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS | {"authorization"}
    }


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }


def create_app(api_key_sha256: bytes) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(connect=5, read=None, write=60, pool=5)
        limits = httpx.Limits(max_connections=1024, max_keepalive_connections=128)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            app.state.client = client
            yield

    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        if not _authorized(request, api_key_sha256):
            return JSONResponse(
                {"error": {"message": "Unauthorized", "type": "authentication_error"}},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        data = []
        for upstream in MODEL_UPSTREAMS.values():
            response = await request.app.state.client.get(f"{upstream}/v1/models")
            response.raise_for_status()
            data.extend(response.json()["data"])
        return JSONResponse({"object": "list", "data": data})

    @app.get("/v1/models/{model}")
    async def model(model: str, request: Request) -> Response:
        if not _authorized(request, api_key_sha256):
            return JSONResponse(
                {"error": {"message": "Unauthorized", "type": "authentication_error"}},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        upstream = MODEL_UPSTREAMS.get(model)
        if upstream is None:
            return _error(f"Unknown model: {model}", 404)
        response = await request.app.state.client.get(f"{upstream}/v1/models/{model}")
        return Response(
            response.content,
            status_code=response.status_code,
            headers=_response_headers(response),
        )

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy(path: str, request: Request) -> Response:
        if not _authorized(request, api_key_sha256):
            return JSONResponse(
                {"error": {"message": "Unauthorized", "type": "authentication_error"}},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > MAX_REQUEST_BYTES:
                return _error("Request body is too large", 413)
        body = bytes(chunks)
        try:
            model = json.loads(body).get("model")
        except (json.JSONDecodeError, AttributeError):
            return _error("Request body must be a JSON object", 400)
        if not isinstance(model, str):
            return _error("Request body must contain a string model", 400)
        upstream = MODEL_UPSTREAMS.get(model)
        if upstream is None:
            return _error(f"Unknown model: {model}", 404)

        url = httpx.URL(f"{upstream}/v1/{path}").copy_with(
            query=request.url.query.encode()
        )
        client: httpx.AsyncClient = request.app.state.client
        upstream_request = client.build_request(
            request.method,
            url,
            headers=_forward_headers(request),
            content=body,
        )
        upstream_response = await client.send(upstream_request, stream=True)
        headers = _response_headers(upstream_response)
        if upstream_response.headers.get("content-type", "").startswith(
            "text/event-stream"
        ):
            return StreamingResponse(
                upstream_response.aiter_raw(),
                status_code=upstream_response.status_code,
                headers=headers,
                background=BackgroundTask(upstream_response.aclose),
            )
        content = await upstream_response.aread()
        await upstream_response.aclose()
        return Response(
            content,
            status_code=upstream_response.status_code,
            headers=headers,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-sha256-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    args = parser.parse_args()
    api_key_sha256 = args.api_key_sha256_file.read_bytes()
    if len(api_key_sha256) != hashlib.sha256().digest_size:
        raise ValueError("API key SHA-256 verifier must be exactly 32 bytes")
    uvicorn.run(create_app(api_key_sha256), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
