from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_buckets(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


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
    # Comma-separated length caps (in tokens). A request whose tokenized
    # length is <= the smallest cap that fits goes into that bucket; each
    # bucket batches independently. Defaults sized against the Jigsaw
    # locust sample distribution (58% / 37% / 5% traffic by bucket).
    batching_buckets: str = "64,256,512"

    @property
    def batching_bucket_list(self) -> list[int]:
        return _parse_buckets(self.batching_buckets)


settings = Settings()
