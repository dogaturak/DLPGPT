"""
Global token usage tracker. Thread-safe singleton.
Usage:
    from graph.token_tracker import tracker
    tracker.reset("my_method")
    # ... run experiment ...
    stats = tracker.stats()
"""
from __future__ import annotations
import threading
from typing import Dict


class TokenTracker:
    _lock = threading.Lock()

    def __init__(self):
        self._label: str = ""
        self._prompt: int = 0
        self._completion: int = 0
        self._calls: int = 0
        self._snapshots: Dict[str, dict] = {}
        self.tracking_failed: bool = False

    def reset(self, label: str = "") -> None:
        with self._lock:
            self._label = label
            self._prompt = 0
            self._completion = 0
            self._calls = 0
            self.tracking_failed = False

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self._prompt += prompt_tokens
            self._completion += completion_tokens
            self._calls += 1

    def snapshot(self, label: str | None = None) -> dict:
        """Save current totals under a label and return them."""
        with self._lock:
            lbl = label or self._label
            data = {
                "label": lbl,
                "prompt_tokens": self._prompt,
                "completion_tokens": self._completion,
                "total_tokens": self._prompt + self._completion,
                "calls": self._calls,
            }
            self._snapshots[lbl] = data
            return data

    def all_snapshots(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._snapshots)

    def warn_if_failed(self) -> None:
        if self.tracking_failed:
            import warnings
            warnings.warn(
                "Token tracking was unreliable — "
                "cost/efficiency metrics may be invalid.",
                RuntimeWarning,
                stacklevel=2,
            )


# Global singleton
tracker = TokenTracker()