"""Read-only views over the audit trail.

`audit/inspections.jsonl` is already the source of truth: the orchestrator
appends one JSON object per inspection and nothing ever rewrites it. The history
view and the export both read that file and nothing else - no second store, no
index, no cache to fall out of step with it.

Everything here is pure apart from one file read, so the formatting is unit
testable without a DevKit, and nothing on this path can touch the detector, the
models, the gate or inspection latency.
"""

from __future__ import annotations

import csv
import io
import json
import re
import struct
import zipfile
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

#: How many records the history endpoint will return at most. The file is read
#: with a bounded deque, so memory stays flat however long the trail gets.
DEFAULT_LIMIT = 200
MAX_LIMIT = 2000

#: Inspection ids are hex, so a dot has no business in one. Excluding it keeps
#: ".." from surviving sanitisation as a substring of a packaged filename.
_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def read_records(path: Path, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """The most recent `limit` inspections, newest first.

    A line that cannot be parsed is skipped rather than failing the request: a
    crash mid-write can leave a partial final line, and one torn line must not
    make the whole history unreadable.
    """
    limit = max(1, min(int(limit), MAX_LIMIT))
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        # deque(maxlen=) keeps only the tail in memory regardless of file size.
        for line in deque(fh, maxlen=limit):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    out.reverse()
    return out


def select_by_ids(records: list[dict], ids: list[str] | None) -> list[dict]:
    """Keep only the records whose id was asked for, in the order given.

    Ids are matched against the audit trail rather than trusted: an id that is
    not in the trail simply selects nothing, so a caller can never reach a record
    - or an evidence file - that the trail does not already contain. That is what
    keeps the export endpoints from becoming an arbitrary file reader.
    """
    if not ids:
        return records
    wanted = [i for i in dict.fromkeys(ids) if i]
    by_id = {str(r.get("id", "")): r for r in records}
    return [by_id[i] for i in wanted if i in by_id]


def parse_ids(raw: str | None) -> list[str]:
    """Comma-separated inspection ids from a query string, sanitised.

    Only the characters an id can actually contain survive, so nothing
    path-shaped ever reaches the filesystem.
    """
    if not raw:
        return []
    return [_SAFE.sub("", part.strip())[:64] for part in raw.split(",") if part.strip()][:MAX_LIMIT]


def iso(ts: float | None) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(float(ts), tz=UTC).isoformat(timespec="seconds")


def _objects_summary(objects: list[dict]) -> str:
    """"person 49/49 (100%)" for each object, which is the whole grounding story
    in one cell."""
    parts = []
    for o in objects or []:
        present = o.get("frames_present", 0)
        total = o.get("frames_total", 0)
        pct = round(float(o.get("presence_ratio", 0.0)) * 100)
        parts.append(f"{o.get('label', '?')} {present}/{total} ({pct}%)")
    return "; ".join(parts)


def summarise(rec: dict) -> dict:
    """One compact row for the history table.

    Deliberately flattens only what a person reads at a glance; the full record
    stays available in the raw file and in the row's own detail.
    """
    metrics = rec.get("metrics") or {}
    window = rec.get("window") or {}
    vlm = rec.get("vlm") or {}
    return {
        "id": rec.get("id", ""),
        "ts": rec.get("ts", 0.0),
        "backend": rec.get("backend", "unknown"),
        "iso": iso(rec.get("ts")),
        "verdict": rec.get("verdict", "unclear"),
        "decided_by": rec.get("decided_by", ""),
        "standard": rec.get("standard", ""),
        # Empty for a single-rule record and for anything written before
        # multi-rule existed, which is what the console keys off to decide
        # whether a per-rule breakdown is worth showing at all.
        "rules": [r for r in (rec.get("rules") or []) if isinstance(r, dict)]
                 if len(rec.get("rules") or []) > 1 else [],
        "reason": rec.get("reason", ""),
        "required": rec.get("required_objects") or [],
        "prohibited": rec.get("prohibited_objects") or [],
        "required_summary": _objects_summary(rec.get("required_objects")),
        "prohibited_summary": _objects_summary(rec.get("prohibited_objects")),
        "window_frames": window.get("total_frames"),
        "window_s": window.get("duration_s"),
        "vlm_ms": metrics.get("inference_ms"),
        "end_to_end_ms": metrics.get("end_to_end_ms"),
        "frames_judged": vlm.get("frames_judged"),
        "trigger_label": rec.get("trigger_label"),
        "frames": [
            {"path": f.get("path"), "rel_ts": f.get("rel_ts"),
             "detections": f.get("detections") or []}
            for f in (rec.get("frames") or []) if f.get("path")
        ] or ([{"path": rec["evidence_path"], "rel_ts": None, "detections": []}]
              if rec.get("evidence_path") else []),
    }


#: Export columns. Ordered for a spreadsheet: when, what was asked, what was
#: decided, why, what the detector actually saw, and how long it took.
CSV_COLUMNS = (
    "timestamp_iso",
    "timestamp_unix",
    "verdict",
    "decided_by",
    "standard",
    "reason",
    # Two is the supported maximum, so explicit columns beat a packed field:
    # they survive a spreadsheet, sort, and filter the way a reader expects.
    # A single-rule inspection leaves the rule_2 columns empty.
    "overall_verdict",
    "rule_1",
    "rule_1_verdict",
    "rule_1_reason",
    "rule_2",
    "rule_2_verdict",
    "rule_2_reason",
    "required_objects",
    "prohibited_objects",
    "window_frames",
    "window_seconds",
    "vlm_ms",
    "end_to_end_ms",
    "trigger_label",
    "backend",
    "evidence_frames",
    "evidence_files",
    "evidence_missing",
    "inspection_id",
)


def _rule_field(rules: list[dict], i: int, key: str) -> str:
    """One field of rule `i`, or "" when that rule does not exist."""
    return str(rules[i].get(key, "")) if i < len(rules) else ""


def to_csv(records: list[dict], evidence_dir: Path | None = None) -> str:
    """Render records as CSV. Reads only; the audit file is never touched.

    Evidence is carried as filenames, never as embedded image data: base64 in a
    CSV cell would inflate the file by a third, break readability and defeat
    every tool that would otherwise open it.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for rec in records:
        row = summarise(rec)
        manifest = evidence_manifest(rec, evidence_dir)
        # A record written before multi-rule existed has no "rules" list; treat
        # it as the single rule it was, so old exports keep their columns filled.
        rules = rec.get("rules") or [{
            "text": rec.get("standard", ""),
            "verdict": rec.get("verdict", ""),
            "reason": rec.get("reason", ""),
        }]
        writer.writerow({
            "timestamp_iso": row["iso"],
            "timestamp_unix": round(float(row["ts"]), 3) if row["ts"] else "",
            "verdict": row["verdict"],
            "decided_by": row["decided_by"],
            "standard": row["standard"],
            "reason": row["reason"],
            "overall_verdict": row["verdict"],
            "rule_1": _rule_field(rules, 0, "text"),
            "rule_1_verdict": _rule_field(rules, 0, "verdict"),
            "rule_1_reason": _rule_field(rules, 0, "reason"),
            "rule_2": _rule_field(rules, 1, "text"),
            "rule_2_verdict": _rule_field(rules, 1, "verdict"),
            "rule_2_reason": _rule_field(rules, 1, "reason"),
            "required_objects": row["required_summary"],
            "prohibited_objects": row["prohibited_summary"],
            "window_frames": row["window_frames"] if row["window_frames"] is not None else "",
            "window_seconds": row["window_s"] if row["window_s"] is not None else "",
            "vlm_ms": row["vlm_ms"] if row["vlm_ms"] is not None else "",
            "end_to_end_ms": row["end_to_end_ms"] if row["end_to_end_ms"] is not None else "",
            "trigger_label": row["trigger_label"] or "",
            "backend": row["backend"],
            "evidence_frames": len(row["frames"]),
            "evidence_files": ";".join(e["file"] for e in manifest),
            "evidence_missing": sum(1 for e in manifest if e["missing"]),
            "inspection_id": row["id"],
        })
    return buf.getvalue()


# --------------------------------------------------------------- evidence

#: Inspection ids are hex, so a dot has no business in one. Excluding it keeps
#: ".." from surviving sanitisation as a substring of the packaged filename.


def safe_evidence_path(evidence_dir: Path, rel: str) -> Path | None:
    """Resolve an evidence reference inside `evidence_dir`, or None.

    The audit trail is written by this application, but it is still a file on
    disk that something else could edit. Resolving and then checking containment
    means a crafted "../../etc/passwd" cannot be read into an export.
    """
    if not rel:
        return None
    base = evidence_dir.resolve()
    candidate = (base / Path(rel).name).resolve()
    if base != candidate.parent:
        return None
    return candidate if candidate.is_file() else None


def package_name(rec: dict, index: int, source: str) -> str:
    """A deterministic, collision-safe name for one evidence file in the package.

    `<utc-compact>-<inspection-id>-<n>.<ext>`: sorted chronologically in a file
    listing, unique because the inspection id is, and stable across re-exports
    of the same record.
    """
    ts = rec.get("ts") or 0.0
    stamp = datetime.fromtimestamp(float(ts), tz=UTC).strftime("%Y%m%dT%H%M%S") if ts else "00000000T000000"
    ident = _SAFE.sub("", str(rec.get("id") or "unknown"))[:24] or "unknown"
    ext = (Path(source).suffix or ".jpg").lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    return f"{stamp}-{ident}-{index + 1}{ext}"


def jpeg_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a JPEG's SOF marker, without decoding the image.

    Stdlib only on purpose: the host has no imaging dependency and an export
    should not introduce one.
    """
    try:
        with path.open("rb") as fh:
            if fh.read(2) != b"\xff\xd8":
                return None
            while True:
                marker = fh.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                kind = marker[1]
                (length,) = struct.unpack(">H", fh.read(2))
                # SOF0..SOF15, excluding the non-frame markers in that range
                if 0xC0 <= kind <= 0xCF and kind not in (0xC4, 0xC8, 0xCC):
                    fh.read(1)
                    height, width = struct.unpack(">HH", fh.read(4))
                    return width, height
                fh.seek(length - 2, 1)
    except (OSError, struct.error):
        return None


def evidence_manifest(rec: dict, evidence_dir: Path | None) -> list[dict]:
    """Structured evidence for one record: never base64, always references.

    A frame whose file is gone is still listed, flagged `missing`, so a lost
    image degrades one row of the report instead of failing the whole export.
    """
    out: list[dict] = []
    for i, frame in enumerate(summarise(rec)["frames"]):
        source = str(frame.get("path") or "")
        entry: dict = {
            "file": package_name(rec, i, source),
            "source": source,
            "rel_ts": frame.get("rel_ts"),
            "detections": frame.get("detections") or [],
            "missing": True,
        }
        resolved = safe_evidence_path(evidence_dir, source) if evidence_dir else None
        if resolved is not None:
            entry["missing"] = False
            entry["bytes"] = resolved.stat().st_size
            size = jpeg_size(resolved)
            if size:
                entry["width"], entry["height"] = size
        out.append(entry)
    return out


def to_json(records: list[dict], evidence_dir: Path | None = None) -> str:
    """The report as JSON, with evidence as structured references."""
    payload = {
        "exported_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "count": len(records),
        "records": [
            summarise(r) | {"evidence": evidence_manifest(r, evidence_dir)}
            for r in records
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_package(records: list[dict], evidence_dir: Path | None) -> bytes:
    """A self-contained .zip: report.csv, report.json and the evidence JPEGs.

    Only evidence belonging to the exported records is included, copied byte for
    byte from the originals. Nothing in the audit trail is read for writing or
    modified.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.csv", to_csv(records, evidence_dir))
        zf.writestr("report.json", to_json(records, evidence_dir))
        seen: set[str] = set()
        missing: list[str] = []
        for rec in records:
            for entry in evidence_manifest(rec, evidence_dir):
                if entry["missing"]:
                    missing.append(f"{entry['file']}  (source: {entry['source']})")
                    continue
                if entry["file"] in seen:
                    continue
                resolved = safe_evidence_path(evidence_dir, entry["source"])
                if resolved is None:
                    continue
                zf.write(resolved, f"evidence/{entry['file']}")
                seen.add(entry["file"])
        if missing:
            zf.writestr(
                "evidence/MISSING.txt",
                "These evidence frames are referenced by the report but were not\n"
                "found on disk at export time. The inspections themselves are still\n"
                "included in report.csv and report.json.\n\n" + "\n".join(missing) + "\n",
            )
    return buf.getvalue()
