from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_signal_descriptions import (
    CATEGORICAL_FIELDS,
    DEFAULT_TOLERANCES,
    NUMERIC_FIELDS,
    consistency_violations,
    finite_float,
    numeric_error,
    numeric_field_applicable,
)


DEFAULT_DISPLAY_FIELDS = (
    "los_status",
    "first_path_delay_ns",
    "delay_spread_ns",
    "k_factor_db",
    "path_count",
)

FIELD_LABELS = {
    "environment": "environment",
    "los_status": "LoS state",
    "path_count": "paths",
    "first_path_delay_ns": "first delay",
    "first_path_angle_deg": "first angle",
    "first_path_power_dbw": "first power",
    "k_factor_db": r"$K$-factor",
    "delay_spread_ns": "delay spread",
    "angle_spread_deg": "azimuth spread",
    "los_delay_ns": "LoS delay",
    "los_angle_deg": "LoS angle",
    "reflection_count": "reflection interactions",
    "reflection_path_count": "reflected paths",
}

FIELD_UNITS = {
    "first_path_delay_ns": "ns",
    "first_path_angle_deg": "deg",
    "first_path_power_dbw": "dBW",
    "k_factor_db": "dB",
    "delay_spread_ns": "ns",
    "angle_spread_deg": "deg",
    "los_delay_ns": "ns",
    "los_angle_deg": "deg",
}

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
}
COUNT_PATTERN = rf"(?:{NUMBER_PATTERN}|{'|'.join(NUMBER_WORDS)})"
TEXT_FIELD_PATTERNS = {
    "path_count": (
        rf"\bwith\s+(?P<value>{COUNT_PATTERN})\s+(?:resolvable\s+|propagation\s+)?paths?\b",
        rf"\bpath\s+count\s+(?:is|of|=)\s*(?P<value>{COUNT_PATTERN})\b",
        rf"\b(?P<value>{COUNT_PATTERN})\s+(?:resolvable\s+|propagation\s+)?paths?\b",
    ),
    "first_path_delay_ns": (
        rf"\bfirst[- ]path\s+delay\s+(?:is|of|=)\s*(?P<value>{NUMBER_PATTERN})\s*ns\b",
    ),
    "first_path_angle_deg": (
        rf"\bfirst[- ]path\s+angle\s+(?:is|of|=)\s*(?P<value>{NUMBER_PATTERN})\s*(?:deg|degrees?)\b",
    ),
    "first_path_power_dbw": (
        rf"\bfirst[- ]path\s+power\s+(?:is|of|=)\s*(?P<value>{NUMBER_PATTERN})\s*dBW\b",
    ),
    "k_factor_db": (
        rf"\bK[- ]factor\s+(?:is|of|=)\s*(?P<value>{NUMBER_PATTERN})\s*dB\b",
    ),
    "delay_spread_ns": (
        rf"\b(?:RMS\s+)?delay\s+spread\s+(?:is|of|=)\s*(?:about\s+|approximately\s+)?(?P<value>{NUMBER_PATTERN})\s*ns\b",
    ),
    "angle_spread_deg": (
        rf"\b(?:azimuth\s+)?angle\s+spread\s+(?:is|of|=)\s*(?P<value>{NUMBER_PATTERN})\s*(?:deg|degrees?)\b",
    ),
    "los_delay_ns": (
        rf"\bLoS\s+delay\s+(?:is|of|=)\s*(?P<value>{NUMBER_PATTERN})\s*ns\b",
    ),
    "los_angle_deg": (
        rf"\bLoS\s+angle\s+(?:is|of|=)\s*(?P<value>{NUMBER_PATTERN})\s*(?:deg|degrees?)\b",
    ),
    "reflection_count": (
        rf"\breflection[- ]interaction\s+count\s+(?:is|of|=)\s*(?P<value>{COUNT_PATTERN})\b",
    ),
    "reflection_path_count": (
        rf"\breflected[- ]path\s+count\s+(?:is|of|=)\s*(?P<value>{COUNT_PATTERN})\b",
    ),
}

STATUS_SYMBOLS = {
    "correct": "✓",
    "out_of_tolerance": "△",
    "incorrect": "✗",
    "hallucinated": "✗",
    "missing": "Missing",
    "not_applicable": "N/A",
}

TEX_STATUS_MACROS = {
    "correct": "qcorrect",
    "out_of_tolerance": "qwarning",
    "incorrect": "qerror",
    "hallucinated": "qerror",
    "missing": "qmissing",
    "not_applicable": "qneutral",
}


@dataclass(frozen=True)
class PayloadSample:
    position: int
    index: int
    group_id: str
    config_key: str
    predicted_record: dict[str, Any]
    target_record: dict[str, Any]
    predicted_text: str
    target_text: str


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("A direct-LLM input name cannot be empty.")
    return name, Path(raw_path)


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary payload in {path}.")
    required = (
        "predicted_signal_records",
        "target_signal_records",
        "predicted_signal_descriptions",
        "target_signal_descriptions",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Payload {path} is missing: {', '.join(missing)}")
    lengths = {key: len(payload[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Payload {path} has inconsistent list lengths: {lengths}")
    return payload


def payload_samples(payload: dict[str, Any]) -> list[PayloadSample]:
    comparisons = payload.get("comparisons")
    count = len(payload["predicted_signal_records"])
    if not isinstance(comparisons, list) or len(comparisons) != count:
        comparisons = [{} for _ in range(count)]

    samples = []
    seen_indices: set[int] = set()
    for position in range(count):
        comparison = comparisons[position]
        if not isinstance(comparison, dict):
            comparison = {}
        index = int(comparison.get("index", position))
        if index in seen_indices:
            raise ValueError(f"Duplicate sample index {index} in payload.")
        seen_indices.add(index)
        samples.append(
            PayloadSample(
                position=position,
                index=index,
                group_id=str(comparison.get("group_id", "")),
                config_key=str(comparison.get("config_key", "")),
                predicted_record=dict(payload["predicted_signal_records"][position]),
                target_record=dict(payload["target_signal_records"][position]),
                predicted_text=str(payload["predicted_signal_descriptions"][position]),
                target_text=str(payload["target_signal_descriptions"][position]),
            )
        )
    return samples


def records_compatible(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    atol: float = 1.0e-4,
) -> bool:
    for field in (*CATEGORICAL_FIELDS, *NUMERIC_FIELDS):
        first_number = finite_float(first.get(field))
        second_number = finite_float(second.get(field))
        if first_number is not None or second_number is not None:
            if first_number is None or second_number is None:
                return False
            if numeric_error(field, first_number, second_number) > atol:
                return False
        elif str(first.get(field, "")).lower() != str(second.get(field, "")).lower():
            return False
    return True


def align_payloads(
    ours: list[PayloadSample],
    direct: list[PayloadSample],
) -> list[tuple[PayloadSample, PayloadSample]]:
    def identity(sample: PayloadSample) -> str | None:
        group_id = sample.group_id.strip()
        if not group_id:
            return None
        return group_id

    def unique_identity_map(
        samples: list[PayloadSample],
    ) -> dict[str, PayloadSample]:
        grouped: dict[str, list[PayloadSample]] = {}
        for sample in samples:
            key = identity(sample)
            if key is not None:
                grouped.setdefault(key, []).append(sample)
        return {key: values[0] for key, values in grouped.items() if len(values) == 1}

    ours_by_identity = unique_identity_map(ours)
    direct_by_identity = unique_identity_map(direct)
    common_identities = set(ours_by_identity) & set(direct_by_identity)
    if common_identities:
        aligned = []
        for ours_sample in ours:
            key = identity(ours_sample)
            if key not in common_identities:
                continue
            direct_sample = direct_by_identity[key]
            if not records_compatible(ours_sample.target_record, direct_sample.target_record):
                raise ValueError(
                    "Target records differ for matching group identity "
                    f"{key!r}; verify that both payloads use the same label version."
                )
            aligned.append((ours_sample, direct_sample))
        return aligned

    if ours_by_identity and direct_by_identity:
        raise ValueError(
            "The payloads contain group identities but have no shared samples; "
            "evaluate both models on the same test cohort."
        )

    direct_by_index = {sample.index: sample for sample in direct}
    aligned = []
    for ours_sample in ours:
        direct_sample = direct_by_index.get(ours_sample.index)
        if direct_sample is None:
            continue
        if not records_compatible(ours_sample.target_record, direct_sample.target_record):
            raise ValueError(
                "Target records differ after alignment at sample index "
                f"{ours_sample.index}; use payloads from the same evaluation cohort."
            )
        aligned.append((ours_sample, direct_sample))
    if not aligned:
        raise ValueError(
            "The proposed-model and direct-LLM payloads have no common sample indices."
        )
    return aligned


def field_assessment(
    field: str,
    prediction: Any,
    target: Any,
    tolerances: dict[str, float],
) -> dict[str, Any]:
    if field in CATEGORICAL_FIELDS:
        target_present = target is not None and str(target).strip() != ""
        prediction_present = prediction is not None and str(prediction).strip() != ""
        if not target_present and not prediction_present:
            status = "not_applicable"
        elif not prediction_present:
            status = "missing"
        elif not target_present:
            status = "hallucinated"
        elif str(prediction).strip().lower() == str(target).strip().lower():
            status = "correct"
        else:
            status = "incorrect"
        return {
            "field": field,
            "status": status,
            "prediction": prediction,
            "target": target,
            "error": None,
            "tolerance": None,
        }

    prediction_value = finite_float(prediction)
    target_value = finite_float(target)
    if target_value is None and prediction_value is None:
        status = "not_applicable"
        error = None
    elif prediction_value is None:
        status = "missing"
        error = None
    elif target_value is None:
        status = "hallucinated"
        error = None
    else:
        error = numeric_error(field, prediction_value, target_value)
        status = "correct" if error <= tolerances[field] else "out_of_tolerance"
    return {
        "field": field,
        "status": status,
        "prediction": prediction_value,
        "target": target_value,
        "error": error,
        "tolerance": tolerances.get(field),
    }


def assess_record(
    prediction: dict[str, Any],
    target: dict[str, Any],
    tolerances: dict[str, float],
) -> dict[str, dict[str, Any]]:
    assessments = {}
    for field in (*CATEGORICAL_FIELDS, *NUMERIC_FIELDS):
        applicable = numeric_field_applicable(field, target) if field in NUMERIC_FIELDS else True
        if not applicable:
            assessments[field] = {
                "field": field,
                "status": "not_applicable",
                "prediction": prediction.get(field),
                "target": target.get(field),
                "error": None,
                "tolerance": tolerances.get(field),
            }
            continue
        assessments[field] = field_assessment(
            field,
            prediction.get(field),
            target.get(field),
            tolerances,
        )
    return assessments


def extract_text_field(
    field: str,
    text: str,
    record_hint: dict[str, Any],
) -> tuple[Any, tuple[int, int] | None, str | None]:
    if field == "los_status":
        explicit_nlos = re.search(r"\bNLoS\b", text, flags=re.IGNORECASE)
        if explicit_nlos is not None:
            return "nlos", explicit_nlos.span(), explicit_nlos.group(0)
        negative = re.search(
            r"\b(?:no\s+direct|without(?:\s+a)?|non[- ]?)\s*LoS\b",
            text,
            flags=re.IGNORECASE,
        )
        if negative is not None:
            los_span = re.search(r"LoS", negative.group(0), flags=re.IGNORECASE)
            assert los_span is not None
            start = negative.start() + los_span.start()
            end = negative.start() + los_span.end()
            return "nlos", (start, end), text[start:end]
        match = re.search(r"\b(?P<value>LoS)\b", text, flags=re.IGNORECASE)
        if match is None:
            return None, None, None
        return "los", match.span("value"), match.group("value")

    if field == "environment":
        value = record_hint.get(field)
        if value is None or not str(value).strip():
            return None, None, None
        match = re.search(rf"\b{re.escape(str(value).strip())}\b", text, flags=re.IGNORECASE)
        if match is None:
            return None, None, None
        return str(value).strip(), match.span(), match.group(0)

    for pattern in TEXT_FIELD_PATTERNS.get(field, ()):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            raw_value = match.group("value")
            numeric_value = NUMBER_WORDS.get(raw_value.lower(), finite_float(raw_value))
            return numeric_value, match.span("value"), raw_value
    return None, None, None


def assess_text(
    text: str,
    record_hint: dict[str, Any],
    target: dict[str, Any],
    tolerances: dict[str, float],
) -> dict[str, dict[str, Any]]:
    assessments = {}
    for field in (*CATEGORICAL_FIELDS, *NUMERIC_FIELDS):
        applicable = numeric_field_applicable(field, target) if field in NUMERIC_FIELDS else True
        value, span, matched_text = extract_text_field(field, text, record_hint)
        if not applicable:
            status = "hallucinated" if value is not None else "not_applicable"
            assessment = {
                "field": field,
                "status": status,
                "prediction": value,
                "target": target.get(field),
                "error": None,
                "tolerance": tolerances.get(field),
            }
        else:
            assessment = field_assessment(
                field,
                value,
                target.get(field),
                tolerances,
            )
        assessment["text_span"] = list(span) if span is not None else None
        assessment["matched_text"] = matched_text
        assessments[field] = assessment
    return assessments


def assessment_counts(assessments: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUS_SYMBOLS}
    for assessment in assessments.values():
        counts[assessment["status"]] += 1
    return counts


def sample_error_score(assessments: dict[str, dict[str, Any]]) -> float:
    score = 0.0
    for assessment in assessments.values():
        status = assessment["status"]
        if status == "correct" or status == "not_applicable":
            continue
        if status == "out_of_tolerance":
            tolerance = float(assessment["tolerance"])
            score += min(float(assessment["error"]) / max(tolerance, 1.0e-12), 5.0)
        elif status == "missing":
            score += 2.0
        else:
            score += 3.0
    return score


def target_complexity(record: dict[str, Any]) -> float:
    score = 0.0
    for field in ("path_count", "reflection_count", "reflection_path_count"):
        value = finite_float(record.get(field))
        if value is not None:
            score += max(value, 0.0)
    delay_spread = finite_float(record.get("delay_spread_ns"))
    if delay_spread is not None:
        score += min(max(delay_spread, 0.0) / 25.0, 10.0)
    return score


def select_examples(
    aligned: list[tuple[PayloadSample, PayloadSample]],
    tolerances: dict[str, float],
) -> list[tuple[str, PayloadSample, PayloadSample]]:
    if len(aligned) < 3:
        raise ValueError("At least three aligned samples are required for qualitative export.")

    scored = []
    for ours, direct in aligned:
        ours_assessments = assess_record(ours.predicted_record, ours.target_record, tolerances)
        scored.append(
            {
                "ours": ours,
                "direct": direct,
                "ours_error": sample_error_score(ours_assessments),
                "complexity": target_complexity(ours.target_record),
            }
        )

    selected_indices: set[int] = set()

    los_candidates = [
        item for item in scored if str(item["ours"].target_record.get("los_status", "")).lower() == "los"
    ]
    typical_pool = los_candidates or scored
    typical = min(
        typical_pool,
        key=lambda item: (
            item["ours_error"],
            item["complexity"],
            item["ours"].index,
        ),
    )
    selected_indices.add(typical["ours"].index)

    nlos_candidates = [
        item
        for item in scored
        if item["ours"].index not in selected_indices
        and str(item["ours"].target_record.get("los_status", "")).lower() == "nlos"
    ]
    fallback_complex_pool = nlos_candidates or [
        item for item in scored if item["ours"].index not in selected_indices
    ]
    successful_nlos = [item for item in nlos_candidates if item["ours_error"] == 0.0]
    complex_pool = successful_nlos or fallback_complex_pool
    complex_nlos = max(
        complex_pool,
        key=lambda item: (
            item["complexity"],
            -item["ours_error"],
            -item["ours"].index,
        ),
    )
    selected_indices.add(complex_nlos["ours"].index)

    challenge_pool = [
        item for item in scored if item["ours"].index not in selected_indices
    ]
    challenging = max(
        challenge_pool,
        key=lambda item: (
            item["ours_error"],
            item["complexity"],
            -item["ours"].index,
        ),
    )

    return [
        ("Typical LoS", typical["ours"], typical["direct"]),
        ("Complex NLoS", complex_nlos["ours"], complex_nlos["direct"]),
        ("Challenging", challenging["ours"], challenging["direct"]),
    ]


def format_value(field: str, value: Any) -> str:
    if field in CATEGORICAL_FIELDS:
        if value is None or str(value).strip() == "":
            return "missing"
        normalized = str(value).strip()
        if field == "los_status":
            return "LoS" if normalized.lower() == "los" else "NLoS"
        return normalized
    number = finite_float(value)
    if number is None:
        return "missing"
    if field in {"path_count", "reflection_count", "reflection_path_count"}:
        text = str(int(round(number)))
    else:
        text = f"{number:.1f}"
    unit = FIELD_UNITS.get(field, "")
    return f"{text} {unit}".strip()


def latex_escape(text: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(text))


def tex_status(status: str, text: str) -> str:
    return rf"\{TEX_STATUS_MACROS[status]}{{{text}}}"


def format_record_tex(
    record: dict[str, Any],
    fields: Iterable[str],
    assessments: dict[str, dict[str, Any]] | None = None,
) -> str:
    parts = []
    for field in fields:
        label = FIELD_LABELS[field]
        value = latex_escape(format_value(field, record.get(field)))
        rendered = f"{label}={value}"
        if assessments is not None:
            rendered = tex_status(assessments[field]["status"], rendered)
        parts.append(rendered)
    return "; ".join(parts)


def annotate_text_tex(
    text: str,
    assessments: dict[str, dict[str, Any]],
    fields: Iterable[str],
) -> str:
    spans: list[tuple[int, int, str]] = []
    for field in fields:
        assessment = assessments[field]
        span = assessment.get("text_span")
        if span is None:
            continue
        start, end = int(span[0]), int(span[1])
        if any(start < other_end and end > other_start for other_start, other_end, _ in spans):
            continue
        spans.append((start, end, assessment["status"]))

    pieces = []
    cursor = 0
    for start, end, status in sorted(spans):
        if start < cursor:
            continue
        pieces.append(latex_escape(text[cursor:start]))
        pieces.append(tex_status(status, latex_escape(text[start:end])))
        cursor = end
    pieces.append(latex_escape(text[cursor:]))

    missing = [
        FIELD_LABELS[field]
        for field in fields
        if assessments[field]["status"] == "missing"
    ]
    if missing:
        pieces.append(
            " "
            + tex_status(
                "missing",
                latex_escape("[Missing: " + ", ".join(missing) + "]"),
            )
        )
    return "".join(pieces)


def audit_summary_tex(assessments: dict[str, dict[str, Any]], fields: Iterable[str]) -> str:
    selected = [assessments[field] for field in fields]
    counts = {status: 0 for status in STATUS_SYMBOLS}
    for assessment in selected:
        counts[assessment["status"]] += 1
    parts = [tex_status("correct", f"{counts['correct']} correct")]
    if counts["out_of_tolerance"]:
        parts.append(tex_status("out_of_tolerance", f"{counts['out_of_tolerance']} over tolerance"))
    if counts["incorrect"] + counts["hallucinated"]:
        parts.append(
            tex_status(
                "incorrect",
                f"{counts['incorrect'] + counts['hallucinated']} wrong/unsupported",
            )
        )
    if counts["missing"]:
        parts.append(tex_status("missing", f"{counts['missing']} missing"))
    return ", ".join(parts)


def build_cases(
    selected: list[tuple[str, PayloadSample, PayloadSample]],
    direct_name: str,
    fields: tuple[str, ...],
    tolerances: dict[str, float],
) -> list[dict[str, Any]]:
    cases = []
    for role, ours, direct in selected:
        ours_assessments = assess_record(ours.predicted_record, ours.target_record, tolerances)
        direct_assessments = assess_record(direct.predicted_record, ours.target_record, tolerances)
        ours_text_assessments = assess_text(
            ours.predicted_text,
            ours.predicted_record,
            ours.target_record,
            tolerances,
        )
        direct_text_assessments = assess_text(
            direct.predicted_text,
            direct.predicted_record,
            ours.target_record,
            tolerances,
        )
        target_violations = consistency_violations(
            ours.target_record,
            los_delay_tolerance_ns=tolerances["los_delay_ns"],
            los_angle_tolerance_deg=tolerances["los_angle_deg"],
        )
        ours_all_violations = consistency_violations(
            ours.predicted_record,
            los_delay_tolerance_ns=tolerances["los_delay_ns"],
            los_angle_tolerance_deg=tolerances["los_angle_deg"],
        )
        direct_all_violations = consistency_violations(
            direct.predicted_record,
            los_delay_tolerance_ns=tolerances["los_delay_ns"],
            los_angle_tolerance_deg=tolerances["los_angle_deg"],
        )
        ours_violations = sorted(set(ours_all_violations) - set(target_violations))
        direct_violations = sorted(set(direct_all_violations) - set(target_violations))
        cases.append(
            {
                "role": role,
                "sample_index": ours.index,
                "group_id": ours.group_id,
                "config_key": ours.config_key,
                "display_fields": list(fields),
                "reference": {
                    "record": ours.target_record,
                    "text": ours.target_text,
                    "consistency_violations": target_violations,
                },
                "ours": {
                    "record": ours.predicted_record,
                    "text": ours.predicted_text,
                    "field_assessments": ours_assessments,
                    "text_field_assessments": ours_text_assessments,
                    "counts": assessment_counts(ours_assessments),
                    "text_counts": assessment_counts(ours_text_assessments),
                    "error_score": sample_error_score(ours_assessments),
                    "text_error_score": sample_error_score(ours_text_assessments),
                    "all_consistency_violations": ours_all_violations,
                    "prediction_only_consistency_violations": ours_violations,
                },
                "direct_llm": {
                    "name": direct_name,
                    "record": direct.predicted_record,
                    "text": direct.predicted_text,
                    "field_assessments": direct_assessments,
                    "text_field_assessments": direct_text_assessments,
                    "counts": assessment_counts(direct_assessments),
                    "text_counts": assessment_counts(direct_text_assessments),
                    "error_score": sample_error_score(direct_assessments),
                    "text_error_score": sample_error_score(direct_text_assessments),
                    "all_consistency_violations": direct_all_violations,
                    "prediction_only_consistency_violations": direct_violations,
                },
            }
        )
    return cases


def render_latex(cases: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    lines = [
        "% Auto-generated by scripts/export_qualitative_channel_reports.py",
        "% Requires: \\usepackage[table]{xcolor}, \\usepackage{booktabs}, \\usepackage{array}, \\usepackage{amssymb}",
        r"\definecolor{qualcorrect}{HTML}{237A57}",
        r"\definecolor{qualwarning}{HTML}{C56A00}",
        r"\definecolor{qualerror}{HTML}{B42318}",
        r"\definecolor{qualmissing}{HTML}{777777}",
        r"\definecolor{qualneutral}{HTML}{555555}",
        r"\providecommand{\qcorrect}[1]{\textcolor{qualcorrect}{\ensuremath{\checkmark}\,#1}}",
        r"\providecommand{\qwarning}[1]{\textcolor{qualwarning}{\ensuremath{\triangle}\,#1}}",
        r"\providecommand{\qerror}[1]{\textcolor{qualerror}{\ensuremath{\times}\,#1}}",
        r"\providecommand{\qmissing}[1]{\textcolor{qualmissing}{[M]\,#1}}",
        r"\providecommand{\qneutral}[1]{\textcolor{qualneutral}{#1}}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Representative CSI-to-text reports. Green/check denotes a correct field within the predefined tolerance, orange/triangle a numerical value outside tolerance, red/cross an incorrect or unsupported field, and gray/[M] a missing field. Selection is deterministic: a low-error LoS sample, the most complex zero-error NLoS sample when available, and the highest-error remaining sample.}",
        r"\label{tab:qualitative-channel-reports}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{@{}p{0.14\textwidth}p{0.82\textwidth}@{}}",
        r"\toprule",
    ]
    for case_index, case in enumerate(cases):
        reference = case["reference"]["record"]
        ours = case["ours"]
        direct = case["direct_llm"]
        heading = (
            f"{latex_escape(case['role'])} "
            f"(sample {case['sample_index']})"
        )
        lines.extend(
            [
                rf"\multicolumn{{2}}{{@{{}}l}}{{\textbf{{{heading}}}}} \\",
                "Reference & " + format_record_tex(reference, fields) + r" \\",
                "Ours: structure & "
                + format_record_tex(ours["record"], fields, ours["field_assessments"])
                + r" \\",
                "Ours: report & "
                + annotate_text_tex(
                    ours["text"],
                    ours["text_field_assessments"],
                    fields,
                )
                + r" \\",
                latex_escape(direct["name"])
                + ": report & "
                + annotate_text_tex(
                    direct["text"],
                    direct["text_field_assessments"],
                    fields,
                )
                + r" \\",
                "Field audit & Ours: "
                + audit_summary_tex(ours["text_field_assessments"], fields)
                + (
                    ", "
                    + tex_status(
                        "incorrect",
                        f"{len(ours['prediction_only_consistency_violations'])} structural contradictions",
                    )
                    if ours["prediction_only_consistency_violations"]
                    else ""
                )
                + "; direct LLM: "
                + audit_summary_tex(direct["text_field_assessments"], fields)
                + (
                    "; "
                    + tex_status(
                        "incorrect",
                        f"{len(direct['prediction_only_consistency_violations'])} structural contradictions",
                    )
                    if direct["prediction_only_consistency_violations"]
                    else ""
                )
                + r" \\",
            ]
        )
        if case_index + 1 < len(cases):
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def render_markdown(cases: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    lines = [
        "# Qualitative CSI-to-Text examples",
        "",
        "Legend: 🟢 correct; 🟠 outside tolerance; 🔴 incorrect/unsupported; ⚪ missing.",
        "",
    ]
    emoji = {
        "correct": "🟢",
        "out_of_tolerance": "🟠",
        "incorrect": "🔴",
        "hallucinated": "🔴",
        "missing": "⚪",
        "not_applicable": "",
    }
    for case in cases:
        lines.extend(
            [
                f"## {case['role']} (sample {case['sample_index']})",
                "",
                "**Reference:** "
                + "; ".join(
                    f"{FIELD_LABELS[field]}={format_value(field, case['reference']['record'].get(field))}"
                    for field in fields
                ),
                "",
            ]
        )
        for key, label in (("ours", "Ours"), ("direct_llm", case["direct_llm"]["name"])):
            result = case[key]
            field_text = "; ".join(
                f"{emoji[result['field_assessments'][field]['status']]} "
                f"{FIELD_LABELS[field]}={format_value(field, result['record'].get(field))}"
                for field in fields
            )
            lines.extend(
                [
                    f"**{label} structured:** {field_text}",
                    "",
                    f"**{label} report:** {result['text']}",
                    "",
                    "**Report audit:** "
                    + "; ".join(
                        f"{emoji[result['text_field_assessments'][field]['status']]} {FIELD_LABELS[field]}"
                        for field in fields
                    ),
                    "",
                ]
            )
    return "\n".join(lines)


def write_field_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fieldnames = (
        "case_role",
        "sample_index",
        "model",
        "layer",
        "field",
        "target",
        "prediction",
        "status",
        "error",
        "tolerance",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            for result_key, model_name in (
                ("ours", "ours"),
                ("direct_llm", case["direct_llm"]["name"]),
            ):
                for layer, assessment_key in (
                    ("structured", "field_assessments"),
                    ("report", "text_field_assessments"),
                ):
                    for field, assessment in case[result_key][assessment_key].items():
                        writer.writerow(
                            {
                                "case_role": case["role"],
                                "sample_index": case["sample_index"],
                                "model": model_name,
                                "layer": layer,
                                "field": field,
                                "target": assessment["target"],
                                "prediction": assessment["prediction"],
                                "status": assessment["status"],
                                "error": assessment["error"],
                                "tolerance": assessment["tolerance"],
                            }
                        )


def parse_fields(value: str) -> tuple[str, ...]:
    fields = tuple(field.strip() for field in value.split(",") if field.strip())
    supported = set(CATEGORICAL_FIELDS) | set(NUMERIC_FIELDS)
    unknown = [field for field in fields if field not in supported]
    if unknown:
        raise ValueError(f"Unsupported display fields: {', '.join(unknown)}")
    if not fields:
        raise ValueError("At least one display field is required.")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select and export three aligned qualitative CSI-to-text cases: "
            "a typical LoS sample, a complex NLoS sample, and a challenging sample."
        )
    )
    parser.add_argument("--ours", required=True, help="Our signal_descriptions.pt payload.")
    parser.add_argument(
        "--direct-llm",
        required=True,
        help="Direct-LLM payload as NAME=PATH, for example InternLM3-8B=.../signal_descriptions.pt.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_DISPLAY_FIELDS),
        help="Comma-separated fields to show in the compact paper table.",
    )
    for field, default in DEFAULT_TOLERANCES.items():
        parser.add_argument(
            "--tolerance-" + field.replace("_", "-"),
            type=float,
            default=default,
            help=f"Correctness tolerance for {field} (default: {default:g}).",
        )
    args = parser.parse_args()

    fields = parse_fields(args.fields)
    tolerances = {
        field: float(getattr(args, "tolerance_" + field))
        for field in DEFAULT_TOLERANCES
    }
    if any(value < 0.0 for value in tolerances.values()):
        raise ValueError("Field tolerances must be non-negative.")

    direct_name, direct_path = parse_named_path(args.direct_llm)
    ours_samples = payload_samples(load_payload(Path(args.ours)))
    direct_samples = payload_samples(load_payload(direct_path))
    aligned = align_payloads(ours_samples, direct_samples)
    selected = select_examples(aligned, tolerances)
    cases = build_cases(selected, direct_name, fields, tolerances)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "qualitative_channel_reports.json"
    csv_path = output_dir / "qualitative_channel_report_fields.csv"
    tex_path = output_dir / "qualitative_channel_reports.tex"
    markdown_path = output_dir / "qualitative_channel_reports.md"

    payload = {
        "metadata": {
            "ours_payload": str(Path(args.ours)),
            "ours_sample_count": len(ours_samples),
            "direct_llm_name": direct_name,
            "direct_llm_payload": str(direct_path),
            "direct_llm_sample_count": len(direct_samples),
            "aligned_sample_count": len(aligned),
            "selection_policy": (
                "lowest-error LoS; highest-complexity zero-error NLoS when available; "
                "highest-error remaining"
            ),
            "display_fields": list(fields),
            "tolerances": tolerances,
        },
        "cases": cases,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_field_csv(csv_path, cases)
    tex_path.write_text(render_latex(cases, fields), encoding="utf-8")
    markdown_path.write_text(render_markdown(cases, fields), encoding="utf-8")

    print(f"saved_qualitative_json={json_path}")
    print(f"saved_qualitative_field_csv={csv_path}")
    print(f"saved_qualitative_latex={tex_path}")
    print(f"saved_qualitative_markdown={markdown_path}")
    print(f"ours_sample_count={len(ours_samples)}")
    print(f"direct_llm_sample_count={len(direct_samples)}")
    print(f"aligned_sample_count={len(aligned)}")
    for case in cases:
        print(
            "selected_qualitative_case="
            + json.dumps(
                {
                    "role": case["role"],
                    "sample_index": case["sample_index"],
                    "ours_error_score": case["ours"]["error_score"],
                    "direct_llm_error_score": case["direct_llm"]["error_score"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
