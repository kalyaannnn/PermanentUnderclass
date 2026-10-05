"""Synchronous JSONL sink for low-volume development events."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._stream: TextIO | None = None

    def __enter__(self) -> JsonlLogger:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")
        return self

    def emit(self, event: str, **fields: Any) -> None:
        if self._stream is None:
            raise RuntimeError("logger is not open")
        if {"event", "timestamp"} & fields.keys():
            raise ValueError("reserved log fields")
        record = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **fields}
        self._stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._stream.flush()

    def __exit__(self, *args: object) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
