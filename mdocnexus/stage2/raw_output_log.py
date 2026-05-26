"""Raw compiler output JSONL logging for Stage 2 auditability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RawCompilerOutputLogEntry:
    doc_id: str
    page_index: int
    provider: str
    model_name: Optional[str]
    compiler_version: str
    prompt_version: str
    raw_output: Dict[str, Any]
    raw_output_hash: str
    stage: str
    created_at: str
    raw_text: Optional[str] = None
    provider_error_type: Optional[str] = None
    provider_error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hash_raw_output(raw_output: Dict[str, Any]) -> str:
    """Return a stable sha256 hash for a raw output dict."""

    serialized = json.dumps(raw_output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_raw_output_log_entry(
    doc_id: str,
    page_index: int,
    provider: str,
    model_name: Optional[str],
    compiler_version: str,
    prompt_version: str,
    raw_output: Dict[str, Any],
    stage: str = "stage2_compiler",
    raw_text: Optional[str] = None,
    provider_error_type: Optional[str] = None,
    provider_error_message: Optional[str] = None,
) -> RawCompilerOutputLogEntry:
    """Build a raw output log entry without evaluation-only fields or secrets."""

    return RawCompilerOutputLogEntry(
        doc_id=doc_id,
        page_index=page_index,
        provider=provider,
        model_name=model_name,
        compiler_version=compiler_version,
        prompt_version=prompt_version,
        raw_output=raw_output,
        raw_output_hash=hash_raw_output(raw_output),
        stage=stage,
        created_at=datetime.now(timezone.utc).isoformat(),
        raw_text=raw_text,
        provider_error_type=provider_error_type,
        provider_error_message=provider_error_message,
    )


def write_raw_output_log(path: str | Path, entry: RawCompilerOutputLogEntry) -> None:
    """Append one raw compiler output entry to a JSONL file."""

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(entry.to_dict(), ensure_ascii=False))
        file_obj.write("\n")
