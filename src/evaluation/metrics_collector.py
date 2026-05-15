# src/evaluation/metrics_collector.py
"""
Extracts accuracy, F1, timing, and MRM verdict from crew output strings.
Both manual and auto runners call collect_metrics() and pass the result dict
to the report generator and comparison module.
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def collect_metrics(crew_output: Any, elapsed_seconds: float, mode: str) -> dict:
    """
    Parse a CrewAI output object (or plain string) and return a standardised
    metrics dict.

    Args:
        crew_output:      Raw output returned by crew.kickoff()
        elapsed_seconds:  Wall-clock time for the full pipeline
        mode:             "manual" | "auto"

    Returns:
        {
            "mode":            str,
            "accuracy":        float | None,
            "f1_score":        float | None,
            "precision":       float | None,
            "recall":          float | None,
            "auc_roc":         float | None,
            "mrm_verdict":     str,
            "elapsed_seconds": float,
            "raw_output":      str,
            "errors":          list[str],
        }
    """
    raw = _to_str(crew_output)
    errors = _extract_errors(raw)

    # Use all task outputs concatenated for best coverage
    accuracy  = _extract_accuracy(raw)
    f1        = _extract_f1(raw)
    precision = _extract_precision(raw)
    recall    = _extract_recall(raw)
    auc_roc   = _extract_simple(raw, r"auc[\-_]?roc[\s:=*|]+([0-9]+\.?[0-9]*)")

    return {
        "mode":            mode,
        "accuracy":        accuracy,
        "f1_score":        f1,
        "precision":       precision,
        "recall":          recall,
        "auc_roc":         auc_roc,
        "mrm_verdict":     _extract_verdict(raw),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "raw_output":      raw,
        "errors":          errors,
    }


def merge_metrics(modeling_metrics: dict, mrm_metrics: dict) -> dict:
    """Combine per-phase metric dicts into one final dict."""
    merged = {**modeling_metrics}
    merged["mrm_verdict"] = mrm_metrics.get("mrm_verdict", "UNKNOWN")
    merged["elapsed_seconds"] = max(
        modeling_metrics.get("elapsed_seconds", 0),
        mrm_metrics.get("elapsed_seconds", 0),
    )
    return merged


def normalize_metric(value, low: float = 0.0, high: float = 1.0):
    """Clamp a metric value to [low, high] range (handles %-scale outputs)."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value > 1.0:
        value = value / 100.0
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Metric-specific extractors
# ---------------------------------------------------------------------------

def _strip_code_templates(text: str) -> str:
    """Remove f-string templates and print statements to avoid false matches
    like:  print(f'Accuracy: {accuracy:.4f}')  or  print('F1 Score:', round(f1, 4))
    which contain metric names but not actual values."""
    # Remove f-string contents: f'...{var}...'
    text = re.sub(r"f'[^']*\{[^}]+\}[^']*'", "", text)
    text = re.sub(r'f"[^"]*\{[^}]+\}[^"]*"', "", text)
    # Remove print(... round(...) ...) patterns  — code not output
    text = re.sub(r"print\s*\([^)]*round\s*\([^)]*\)[^)]*\)", "", text, flags=re.IGNORECASE)
    return text


def _extract_accuracy(text: str):
    """Extract accuracy from multiple output formats."""
    clean = _strip_code_templates(text)
    patterns = [
        # sklearn print output:  Accuracy: 0.8626
        r"(?:^|\n)\s*accuracy\s*[:\s=]+([0-9]+\.?[0-9]*)",
        # Markdown bold:  **Accuracy**: 0.8626  or  **Accuracy** - 0.8626
        r"\*{1,2}\s*accuracy\s*\*{1,2}\s*[:\-]+\s*([0-9]+\.?[0-9]*)",
        # Dash list:  - Accuracy: 0.8626
        r"[-*]\s*accuracy\s*[:\s=]+([0-9]+\.?[0-9]*)",
        # Generic inline
        r"accuracy[\s:=|]+([0-9]+\.?[0-9]*)",
        # Classification report table row: accuracy   0.86  230
        r"^\s*accuracy\s+([0-9]+\.[0-9]+)\s+[0-9]+",
        # Baseline / test accuracy
        r"(?:test|baseline|model)\s+accuracy[\s:=*|]+([0-9]+\.?[0-9]*)",
        # Achieved accuracy of X.XXXX
        r"achieved\s+(?:an?\s+)?accuracy\s+of\s+([0-9]+\.?[0-9]*)",
    ]
    for p in patterns:
        v = _extract_simple(clean, p)
        if v is not None and v > 0.0:
            return v
    return None


def _extract_f1(text: str):
    """Extract F1 score from multiple output formats."""
    clean = _strip_code_templates(text)
    patterns = [
        # sklearn print output:  F1 Score: 0.8321  or  F1: 0.8321
        r"(?:^|\n)\s*f1[\-_\s]?score\s*[:\s=]+([0-9]+\.?[0-9]*)",
        r"(?:^|\n)\s*f1\s*[:\s=]+([0-9]+\.?[0-9]*)",
        # Markdown bold
        r"\*{1,2}\s*f1[\-_\s]?score\s*\*{1,2}\s*[:\-]+\s*([0-9]+\.?[0-9]*)",
        r"\*{1,2}\s*f1\s*\*{1,2}\s*[:\-]+\s*([0-9]+\.?[0-9]*)",
        # Dash list
        r"[-*]\s*f1[\-_\s]?score\s*[:\s=]+([0-9]+\.?[0-9]*)",
        # Generic
        r"f1[\-_\s]?score[\s:=|]+([0-9]+\.?[0-9]*)",
        r"\bf1[\s:=|]+([0-9]+\.?[0-9]*)",
        # sklearn classification_report weighted avg:
        # weighted avg  precision  recall  f1-score  support
        # Values appear as:  weighted avg   0.86   0.86   0.86   230
        r"weighted\s+avg\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+([0-9]+\.[0-9]+)",
        r"macro\s+avg\s+[0-9]+\.[0-9]+\s+[0-9]+\.[0-9]+\s+([0-9]+\.[0-9]+)",
    ]
    for p in patterns:
        v = _extract_simple(clean, p)
        if v is not None and v > 0.0:
            return v
    return None


def _extract_precision(text: str):
    """Extract precision from multiple output formats."""
    clean = _strip_code_templates(text)
    patterns = [
        # sklearn print output
        r"(?:^|\n)\s*precision\s*[:\s=]+([0-9]+\.?[0-9]*)",
        # Markdown bold
        r"\*{1,2}\s*precision\s*\*{1,2}\s*[:\-]+\s*([0-9]+\.?[0-9]*)",
        # Dash list
        r"[-*]\s*precision\s*[:\s=]+([0-9]+\.?[0-9]*)",
        # Generic
        r"precision[\s:=|]+([0-9]+\.?[0-9]*)",
        # Classification report weighted avg: precision is 1st numeric column
        r"weighted\s+avg\s+([0-9]+\.[0-9]+)",
        r"macro\s+avg\s+([0-9]+\.[0-9]+)",
    ]
    for p in patterns:
        v = _extract_simple(clean, p)
        if v is not None and v > 0.0:
            return v
    return None


def _extract_recall(text: str):
    """Extract recall from multiple output formats."""
    clean = _strip_code_templates(text)
    patterns = [
        # sklearn print output
        r"(?:^|\n)\s*recall\s*[:\s=]+([0-9]+\.?[0-9]*)",
        # Markdown bold
        r"\*{1,2}\s*recall\s*\*{1,2}\s*[:\-]+\s*([0-9]+\.?[0-9]*)",
        # Dash list
        r"[-*]\s*recall\s*[:\s=]+([0-9]+\.?[0-9]*)",
        # Generic
        r"recall[\s:=|]+([0-9]+\.?[0-9]*)",
        # Classification report weighted avg: recall is 2nd numeric column
        r"weighted\s+avg\s+[0-9]+\.[0-9]+\s+([0-9]+\.[0-9]+)",
        r"macro\s+avg\s+[0-9]+\.[0-9]+\s+([0-9]+\.[0-9]+)",
    ]
    for p in patterns:
        v = _extract_simple(clean, p)
        if v is not None and v > 0.0:
            return v
    return None


def _extract_simple(text: str, pattern: str):
    """Run a single regex and normalize the first capture group."""
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    try:
        return normalize_metric(float(m.group(1)))
    except (ValueError, TypeError):
        return None


def _extract_verdict(text: str) -> str:
    text_upper = text.upper()
    if "APPROVED" in text_upper:
        return "APPROVED"
    if "REJECTED" in text_upper:
        return "REJECTED"
    if "CONDITIONAL" in text_upper:
        return "CONDITIONAL"
    return "UNKNOWN"


def _extract_errors(text: str) -> list:
    errors = []
    for line in text.splitlines():
        if any(kw in line.lower() for kw in ("error", "exception", "traceback", "failed")):
            errors.append(line.strip())
    return errors[:10]


# ---------------------------------------------------------------------------
# Public string converter
# ---------------------------------------------------------------------------

def _to_str(output: Any) -> str:
    if output is None:
        return ""
    if hasattr(output, "tasks_output") and output.tasks_output:
        # Concatenate ALL task outputs for maximum metric coverage
        parts = [str(t.raw) for t in output.tasks_output if t and t.raw]
        return "\n\n---TASK OUTPUT---\n".join(parts)
    if hasattr(output, "raw"):
        return str(output.raw)
    if hasattr(output, "result"):
        return str(output.result)
    return str(output)