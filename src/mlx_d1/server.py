"""FastAPI app. The model is loaded once during startup and then kept resident."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlx_d1 import __version__
from mlx_d1.config import Settings
from mlx_d1.errors import RequestError
from mlx_d1.factory import load_backend
from mlx_d1.schemas import BatchRequest, CompareRequest, PresetRunRequest, SystemOneRequest
from mlx_d1.service import InferenceGate, JudgeService

log = logging.getLogger("mlx_d1")


def create_app(settings: Settings, backend: Any | None = None) -> FastAPI:
    """Build the app. Pass ``backend`` to skip loading weights (tests, fake mode)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = getattr(app.state, "backend", None)
        if active is None:
            log.info("loading decision model %s", app.state.settings.model)
            active = load_backend(app.state.settings)
            app.state.backend = active
            log.info("decision model ready backend=%s", getattr(active, "name", "unknown"))
        app.state.service = JudgeService(app.state.settings, active, InferenceGate())
        try:
            yield
        finally:
            close = getattr(active, "close", None)
            if close is not None:
                close()

    app = FastAPI(
        title="mlx-d1",
        version=__version__,
        description=(
            "Resident decision server for LiquidAI d1. "
            "The request supplies the state, images, and questions. "
            "POST /v1/systemone follows the llama.cpp systemone shape."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    if backend is not None:
        app.state.backend = backend

    @app.exception_handler(RequestError)
    async def on_request_error(_request: Request, exc: RequestError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status, content={"error": {"type": exc.code, "message": str(exc)}}
        )

    @app.exception_handler(RequestValidationError)
    async def on_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": {"type": "validation_error", "message": _format_validation(exc)}},
        )

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "mlx-d1", "health": "/health", "docs": "/docs"}

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        active_backend = request.app.state.backend
        name = getattr(active_backend, "name", "unknown")
        return {
            "status": "ok",
            "model_loaded": name == "mlx",
            "weights_loaded": name == "mlx",
            "backend": name,
            "model": getattr(active_backend, "model_id", request.app.state.settings.model),
            "version": __version__,
        }

    @app.post("/v1/systemone")
    async def systemone(body: SystemOneRequest, request: Request) -> dict[str, Any]:
        return await _service(request).system_one(
            state=body.state,
            images=body.images,
            questions=body.questions,
            score_readout=body.score_readout,
            set_size=body.set_size,
        )

    @app.post("/v1/systemone/batch")
    async def systemone_batch(body: BatchRequest, request: Request) -> dict[str, Any]:
        return await _service(request).system_one_batch(
            [item.model_dump() for item in body.requests],
            score_readout=body.score_readout,
            set_size=body.set_size,
        )

    @app.post("/v1/compare")
    async def compare(body: CompareRequest, request: Request) -> dict[str, Any]:
        return await _service(request).compare(
            a=body.a.model_dump(),
            b=body.b.model_dump(),
            criteria=body.criteria,
            state=body.state,
            score_readout=body.score_readout,
            set_size=body.set_size,
        )

    @app.get("/v1/presets")
    def list_presets(request: Request) -> dict[str, Any]:
        return _service(request).list_presets()

    @app.get("/v1/presets/{name}")
    def get_preset(name: str, request: Request) -> dict[str, Any]:
        return _service(request).get_preset(name)

    @app.post("/v1/presets/{name}")
    async def run_preset(
        name: str,
        request: Request,
        body: PresetRunRequest = Body(default_factory=PresetRunRequest),  # noqa: B008
    ) -> dict[str, Any]:
        return await _service(request).run_preset(
            name,
            state=body.state,
            images=body.images,
            score_readout=body.score_readout,
            set_size=body.set_size,
        )

    return app


def _service(request: Request) -> JudgeService:
    return request.app.state.service


def _format_validation(exc: RequestValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = [str(item) for item in error.get("loc", ()) if item != "body"]
        where = ".".join(location)
        message = str(error.get("msg", "invalid"))
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts) or "invalid request"
