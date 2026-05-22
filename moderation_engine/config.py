from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    model_name: str = "unitary/toxic-bert"
    backend: Literal["pytorch", "onnx"] = "pytorch"
    onnx_model_dir: Path = Path("models/onnx-toxic-bert")
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Dynamic batching. window_ms=0 disables it (each request runs solo) so
    # the same image serves baseline vs batched arms of the sweep.
    batching_window_ms: float = 5.0
    batching_max_batch_size: int = 32


settings = Settings()
