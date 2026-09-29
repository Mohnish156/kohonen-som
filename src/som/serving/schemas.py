"""Request / response contracts for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_created_at: str | None = None
    quantization_error: float | None = None


class MapRequest(BaseModel):
    data: list[list[float]] = Field(description="samples x features", min_length=1)


class MapResponse(BaseModel):
    bmu: list[list[int]] = Field(description="(x, y) grid coordinate of each sample's BMU")


class TrainJobRequest(BaseModel):
    width: int = Field(ge=1, le=500)
    height: int = Field(ge=1, le=500)
    n_epochs: int = Field(default=100, ge=1, le=10_000)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    seed: int | None = None
    data_uri: str | None = Field(default=None, description="where the job should read data from")


class TrainJobResponse(BaseModel):
    job_id: str
    status: str
    detail: str


class DeployRequest(BaseModel):
    artifact_dir: str = Field(description="dir with weights.npz + metadata.json (prod: gs:// URI)")


class DeployResponse(BaseModel):
    status: str
    artifact_dir: str
    detail: str
