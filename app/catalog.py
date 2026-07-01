"""SHL catalog loading, normalization, and indexing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import CATALOG_PATH

KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


@dataclass
class Assessment:
    """A single SHL assessment product."""

    entity_id: str
    name: str
    url: str
    description: str
    keys: list[str]
    test_type_codes: str
    job_levels: list[str]
    languages: list[str]
    duration: str
    duration_minutes: int | None
    remote: bool
    adaptive: bool
    search_text: str = ""

    def to_catalog_entry(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type_codes,
            "keys": self.keys,
            "description": self.description,
            "job_levels": self.job_levels,
            "duration": self.duration,
            "remote": self.remote,
            "adaptive": self.adaptive,
            "languages": self.languages[:5],
        }


def _parse_duration_minutes(duration_raw: str) -> int | None:
    if not duration_raw:
        return None
    match = re.search(r"(\d+)", duration_raw)
    return int(match.group(1)) if match else None


def _derive_test_type_codes(keys: list[str]) -> str:
    codes: list[str] = []
    for key in keys:
        code = KEY_TO_CODE.get(key)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


def _build_search_text(item: dict, test_type_codes: str) -> str:
    parts = [
        f"Assessment: {item['name']}",
        f"Description: {item.get('description', '')}",
        f"Type: {', '.join(item.get('keys', []))}",
        f"Type codes: {test_type_codes}",
        f"Job levels: {', '.join(item.get('job_levels', []))}",
        f"Duration: {item.get('duration', 'Unknown')}",
    ]
    if item.get("adaptive") == "yes":
        parts.append("Adaptive test")
    if item.get("remote") == "yes":
        parts.append("Remote/online assessment")
    return " | ".join(parts)


def load_catalog(catalog_path: str | None = None) -> list[Assessment]:
    """Load and preprocess the SHL catalog JSON."""
    path = Path(catalog_path or CATALOG_PATH)
    with open(path, encoding="utf-8") as f:
        raw_data = json.loads(f.read(), strict=False)

    assessments: list[Assessment] = []
    seen_ids: set[str] = set()

    for item in raw_data:
        entity_id = item.get("entity_id", "")
        if not entity_id or entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)

        if item.get("status") != "ok":
            continue

        keys = item.get("keys", [])
        test_type_codes = _derive_test_type_codes(keys)

        assessment = Assessment(
            entity_id=entity_id,
            name=item.get("name", ""),
            url=item.get("link", ""),
            description=item.get("description", "").replace("\r\n", " ").strip(),
            keys=keys,
            test_type_codes=test_type_codes,
            job_levels=item.get("job_levels", []),
            languages=item.get("languages", []),
            duration=item.get("duration", ""),
            duration_minutes=_parse_duration_minutes(item.get("duration_raw", "")),
            remote=item.get("remote", "no") == "yes",
            adaptive=item.get("adaptive", "no") == "yes",
        )
        assessment.search_text = _build_search_text(item, test_type_codes)
        assessments.append(assessment)

    return assessments


def build_url_index(assessments: list[Assessment]) -> dict[str, Assessment]:
    return {a.url: a for a in assessments}


def build_name_index(assessments: list[Assessment]) -> dict[str, Assessment]:
    return {a.name.lower(): a for a in assessments}
