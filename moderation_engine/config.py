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

    # ONNX Runtime intra-op threading. 0 = ORT default
    # (`num_logical_cores`). Set 1 to force single-threaded ORT so the
    # dynamic batcher can run multiple batches truly in parallel through
    # the executor — the Opt 4 experiment. See `docs/benchmarks.md`
    # "Dynamic batching" for why this matters on a 2-vCPU host.
    onnx_intra_op_threads: int = 0

    # Dynamic batching. Default 0 (disabled) — on the production target
    # (`c6i.large`, 2 vCPU INT8) the EC2 sweep showed bypass mode wins
    # because ORT already saturates both vCPUs at batch=1 (see
    # `docs/benchmarks.md` "Dynamic batching"). The batcher implementation
    # is kept in-tree behind this env var so a larger instance with more
    # vCPUs can flip it on with BATCHING_WINDOW_MS=5 (or any positive int)
    # and pick up the length-bucketed path.
    batching_window_ms: float = 0.0
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
