"""HTTP bodies. Question contents stay untyped so callers can pass every d1 field."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Readout = Literal["median", "mean", "argmax"]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Any = None
    images: list[str] | None = None
    questions: dict[str, Any]
    score_readout: Readout | None = None
    set_size: int | None = Field(default=None, ge=1)


class BatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Any = None
    images: list[str] | None = None
    questions: dict[str, Any]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: list[BatchItem] = Field(min_length=1)
    score_readout: Readout | None = None
    set_size: int | None = Field(default=None, ge=1)


class CompareSide(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    images: list[str] = Field(default_factory=list)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: CompareSide
    b: CompareSide
    criteria: Any
    state: Any = None
    score_readout: Readout | None = None
    set_size: int | None = Field(default=None, ge=1)


class PresetRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Any = None
    images: list[str] | None = None
    score_readout: Readout | None = None
    set_size: int | None = Field(default=None, ge=1)
