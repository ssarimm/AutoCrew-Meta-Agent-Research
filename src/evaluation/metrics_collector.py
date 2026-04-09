import re
import json
from datetime import datetime


class MetricsCollector:
    """
    Extracts and stores metrics from crew execution results.
    Used to compare manual vs auto paths.
    """

    def __init__(self):
        self.records = []

    def extract_metrics(self, results: dict) -> dict:
        """
        Extract quantitative metrics from a pipeline run result.
        
        Args:
            results: dict from manual_runner or auto_runner
            
        Returns:
            dict with extracted metrics
        """
        metrics = {
            "mode": results.get("mode", "unknown"),
            "dataset": results.get("dataset", ""),
            "task": results.get("task", ""),
            "timestamp": datetime.now().isoformat(),
        }

        # Timing metrics
        metrics["meta_agent_time_sec"] = results.get("meta_agent_time_sec", 0)
        metrics["modeling_time_sec"] = results.get("modeling_time_sec", 0)
        metrics["mrm_time_sec"] = results.get("mrm_time_sec", 0)
        metrics["total_time_sec"] = results.get("total_time_sec", 0)

        # Extract ML metrics from modeling output text
        output_text = results.get("modeling_output", "")
        metrics["accuracy"] = self._extract_metric(output_text, "accuracy")
        metrics["f1_score"] = self._extract_metric(output_text, "f1")
        metrics["precision"] = self._extract_metric(output_text, "precision")
        metrics["recall"] = self._extract_metric(output_text, "recall")

        # MRM verdict
        verdict_text = results.get("mrm_verdict", "")
        metrics["mrm_verdict"] = "APPROVED" if "approved" in verdict_text.lower() else "REJECTED"

        # Auto-specific metrics
        if results.get("mode") == "auto":
            config = results.get("generated_config", {})
            modeling = config.get("modeling_crew", {})
            mrm = config.get("mrm_crew", {})
            metrics["auto_modeling_agents"] = len(modeling.get("agents", []))
            metrics["auto_modeling_tasks"] = len(modeling.get("tasks", []))
            metrics["auto_mrm_agents"] = len(mrm.get("agents", []))
            metrics["auto_mrm_tasks"] = len(mrm.get("tasks", []))

        # Manual has fixed counts
        if results.get("mode") == "manual":
            metrics["auto_modeling_agents"] = 3  # hardcoded in base paper
            metrics["auto_modeling_tasks"] = 3
            metrics["auto_mrm_agents"] = 3
            metrics["auto_mrm_tasks"] = 3

        self.records.append(metrics)
        return metrics

    def _extract_metric(self, text: str, metric_name: str) -> float:
        """Extract a numeric metric value from text output."""
        patterns = [
            rf"{metric_name}\s*[:=]\s*([\d.]+)",
            rf"{metric_name}\s*score\s*[:=]\s*([\d.]+)",
            rf"{metric_name}\s*\(.*?\)\s*[:=]\s*([\d.]+)",
        ]

        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    val = float(match.group(1))
                    # If value > 1, it's likely a percentage
                    if val > 1:
                        val = val / 100.0
                    return round(val, 4)
                except ValueError:
                    continue

        return -1.0  # Not found

    def get_all_records(self) -> list:
        return self.records

    def save_records(self, filepath: str = "results/metrics.json"):
        """Save all collected metrics to JSON."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"Metrics saved to {filepath}")
