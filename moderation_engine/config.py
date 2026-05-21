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


settings = Settings()
