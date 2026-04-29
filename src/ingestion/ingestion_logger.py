"""Minimal structured logging for Guardian API usage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.dates import utc_now_checkpoint_token


class IngestionLogger:
    """Write minimal JSONL ingestion events and track API usage counters."""

    def __init__(self, enabled: bool, logs_dir: Path, run_timestamp: Optional[str] = None):
        self._enabled = enabled
        self._logs_dir = logs_dir
        self._run_timestamp = run_timestamp or utc_now_checkpoint_token()
        self._log_path = self._logs_dir / f"ingestion_usage_{self._run_timestamp}.jsonl"
        self._usage_counts: Dict[str, Any] = {
            "total_api_calls": 0,
            "error_api_calls": 0,
            "calls_by_profile": {},
        }

    @property
    def usage_counts(self) -> Dict[str, Any]:
        """Return usage counters collected for the current run."""
        return {
            "total_api_calls": int(self._usage_counts["total_api_calls"]),
            "error_api_calls": int(self._usage_counts["error_api_calls"]),
            "calls_by_profile": dict(self._usage_counts["calls_by_profile"]),
        }

    @property
    def log_path(self) -> Optional[str]:
        """Return the log output path when logging is enabled."""
        if not self._enabled:
            return None
        return str(self._log_path)

    def log_api_call(self, profile: Optional[str], path: str, status_code: Optional[int]) -> None:
        """Record one successful API call."""
        self._increment_usage(profile=profile, is_error=False)
        self._write_event(
            {
                "event": "guardian_api_call",
                "timestamp": utc_now_checkpoint_token(),
                "profile": profile,
                "path": path,
                "status_code": status_code,
                "status": "success",
            }
        )

    def log_api_error(
        self,
        profile: Optional[str],
        path: str,
        error: str,
        status_code: Optional[int],
    ) -> None:
        """Record one failed API call."""
        self._increment_usage(profile=profile, is_error=True)
        self._write_event(
            {
                "event": "guardian_api_error",
                "timestamp": utc_now_checkpoint_token(),
                "profile": profile,
                "path": path,
                "status_code": status_code,
                "status": "error",
                "error": error,
            }
        )

    def log_ingestion_summary(self, payload: Dict[str, Any]) -> None:
        """Record one ingestion summary event."""
        event_payload = dict(payload)
        event_payload["event"] = "ingestion_summary"
        event_payload["timestamp"] = utc_now_checkpoint_token()
        self._write_event(event_payload)

    def _increment_usage(self, profile: Optional[str], is_error: bool) -> None:
        self._usage_counts["total_api_calls"] += 1
        if is_error:
            self._usage_counts["error_api_calls"] += 1
        profile_key = profile or "unknown_profile"
        calls_by_profile = self._usage_counts["calls_by_profile"]
        calls_by_profile[profile_key] = int(calls_by_profile.get(profile_key, 0)) + 1

    def _write_event(self, event: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(event, ensure_ascii=True))
            output_file.write("\n")
