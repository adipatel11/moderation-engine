"""Phase 1 locked load-test protocol for moderation-engine.

Each simulated user is a closed-loop client that posts a random sample drawn
from the Jigsaw scored test split to `POST /classify` with no think time —
this measures the server's pure response-time-under-concurrency curve.

Sampling uses a fixed RNG seed so the sample pool is stable across runs (the
plan requires "same test set" for every benchmark).

Run a single level manually:

    locust -f locustfile.py --headless -u 25 -r 25 --run-time 60s \
        -H http://<host>:8000 --csv docs/locust/u25

Run the whole sweep via `scripts/run_locust_sweep.sh`.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
from locust import HttpUser, between, events, task

ROOT = Path(__file__).resolve().parent
TEST_CSV = ROOT / "data" / "jigsaw" / "test.csv"
LABELS_CSV = ROOT / "data" / "jigsaw" / "test_labels.csv"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

SAMPLE_SIZE = int(os.environ.get("LOCUST_SAMPLE_SIZE", "1000"))
SAMPLE_SEED = int(os.environ.get("LOCUST_SAMPLE_SEED", "42"))
SAMPLE_TEXTS: list[str] = []


@events.init.add_listener
def _load_samples(environment, **_kwargs) -> None:
    """Draw a stable sample of scored rows once at startup."""
    if SAMPLE_TEXTS:
        return
    test = pd.read_csv(TEST_CSV)
    labels = pd.read_csv(LABELS_CSV)
    merged = test.merge(labels, on="id", how="inner")
    scored = merged[(merged[LABEL_COLS] != -1).all(axis=1)].reset_index(drop=True)
    sample = scored.sample(n=min(SAMPLE_SIZE, len(scored)), random_state=SAMPLE_SEED)
    SAMPLE_TEXTS.extend(t for t in sample["comment_text"].fillna("").tolist() if t.strip())
    n_toxic = int(sample[LABEL_COLS].any(axis=1).sum())
    print(
        f"[locustfile] loaded {len(SAMPLE_TEXTS)} samples "
        f"({n_toxic} flagged, {len(SAMPLE_TEXTS) - n_toxic} clean) "
        f"seed={SAMPLE_SEED}"
    )


class ClassifyUser(HttpUser):
    wait_time = between(0.0, 0.0)  # closed-loop: send next as soon as the prior returns

    @task
    def classify(self) -> None:
        text = random.choice(SAMPLE_TEXTS)
        with self.client.post(
            "/classify",
            json={"text": text},
            name="POST /classify",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code} body={resp.text[:200]}")
            elif "labels" not in resp.json():
                resp.failure("missing labels field")
