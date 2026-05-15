import os           # path construction and directory creation
import json          # saving comparison results to disk
import time          # unused here, kept for future timing additions
from datetime import datetime  # timestamping output files

from src.evaluation.metrics_collector import collect_metrics  # extracts numeric metrics from agent text
from src.evaluation.generate_charts import generate_all       # builds charts for both pipelines

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


class ComparisonRunner:
    """Runs both pipelines back-to-back and produces a side-by-side comparison report.
    Path A = Manual (hardcoded agents), Path B = AutoCrew (meta-agent generated)."""

    def run_comparison(self, llm, dataset_path, task_desc, llm_name="ollama"):
        print("\n" + "=" * 60)
        print("  AUTOCREW EVALUATION — COMPARISON MODE")
        print("=" * 60)

        # Lazy imports keep startup fast and avoid circular imports at module level
        print("\n[1/5] Running Manual Pipeline (Path A) …")
        from src.manual.manual_runner import run_manual_pipeline
        manual_results = run_manual_pipeline(llm, dataset_path, task_desc)

        print("\n[2/5] Running AutoCrew Pipeline (Path B) …")
        from src.crew_factory.auto_runner import run_auto_pipeline
        auto_results = run_auto_pipeline(llm, dataset_path, task_desc, llm_name)

        if not manual_results or not auto_results:
            print("\n[ERROR] One or both pipelines failed. Comparison aborted.")
            return None

        print("\n[3/5] Extracting Metrics …")

        def _concat_outputs(raw_results):
            """Merge modeling_output and all_task_texts into one string.
            CrewAI's final output is often the last task only, so we need
            all task outputs to ensure metric lines aren't missed."""
            parts = []
            if raw_results.get("modeling_output"):
                parts.append(str(raw_results["modeling_output"]))
            if raw_results.get("all_task_texts") and isinstance(raw_results["all_task_texts"], list):
                parts.extend([str(t) for t in raw_results["all_task_texts"]])
            return "\n\n---TASK OUTPUT---\n".join(parts) if parts else ""

        dataset_name = os.path.basename(dataset_path)

        manual_metrics = collect_metrics(
            _concat_outputs(manual_results),
            manual_results.get("total_time_sec", 0.0),
            mode="manual"
        )
        manual_metrics["dataset"] = dataset_name
        manual_metrics["mrm_verdict"] = manual_results.get("mrm_verdict", "UNKNOWN")
        manual_metrics["modeling_time_sec"] = manual_results.get("modeling_time_sec", 0.0)
        manual_metrics["mrm_time_sec"] = manual_results.get("mrm_time_sec", 0.0)
        manual_metrics["meta_agent_time_sec"] = 0.0  # manual path has no meta agent overhead
        manual_metrics["total_time_sec"] = manual_results.get("total_time_sec", 0.0)

        auto_metrics = collect_metrics(
            _concat_outputs(auto_results),
            auto_results.get("total_time_sec", 0.0),
            mode="auto"
        )
        auto_metrics["dataset"] = dataset_name
        auto_metrics["mrm_verdict"] = auto_results.get("mrm_verdict", "UNKNOWN")
        auto_metrics["modeling_time_sec"] = auto_results.get("modeling_time_sec", 0.0)
        auto_metrics["mrm_time_sec"] = auto_results.get("mrm_time_sec", 0.0)
        auto_metrics["meta_agent_time_sec"] = auto_results.get("meta_agent_time_sec", 0.0)
        auto_metrics["total_time_sec"] = auto_results.get("total_time_sec", 0.0)

        print("\n[4/5] Generating comparison charts …")
        generate_all(dataset_name=dataset_name, manual_metrics=manual_metrics, auto_metrics=auto_metrics)

        print("\n[5/5] Generating Comparison PDF and saving JSON …")
        from src.evaluation.report_generator import generate_comparison_report
        pdf_c = generate_comparison_report(manual_metrics, auto_metrics)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self._save_json(manual_metrics, auto_metrics, manual_results, auto_results, ts)

        print("\n" + "=" * 60)
        print("  COMPARISON DONE")
        print(f"  Comparison PDF  → {pdf_c}")
        print(f"  Raw JSON        → {json_path}")
        print("=" * 60 + "\n")

        return {
            "manual_metrics": manual_metrics,
            "auto_metrics": auto_metrics,
            "pdf_comparison": pdf_c,
            "json_path": json_path,
        }

    def _save_json(self, manual_metrics, auto_metrics, manual_raw, auto_raw, ts: str) -> str:
        """Save full comparison data to JSON for reproducibility and offline analysis."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"comparison_{ts}.json")

        payload = {
            "timestamp": ts,
            "manual_path": {
                "metrics": manual_metrics,
                "timing": {
                    "modeling_time_sec": manual_raw.get("modeling_time_sec"),
                    "mrm_time_sec": manual_raw.get("mrm_time_sec"),
                    "total_time_sec": manual_raw.get("total_time_sec"),
                },
                "verdict": manual_raw.get("mrm_verdict"),
                "output": manual_raw.get("modeling_output"),
            },
            "auto_path": {
                "metrics": auto_metrics,
                "timing": {
                    "meta_agent_time_sec": auto_raw.get("meta_agent_time_sec"),
                    "modeling_time_sec": auto_raw.get("modeling_time_sec"),
                    "mrm_time_sec": auto_raw.get("mrm_time_sec"),
                    "total_time_sec": auto_raw.get("total_time_sec"),
                },
                "verdict": auto_raw.get("mrm_verdict"),
                "output": auto_raw.get("modeling_output"),
                # The generated config shows what crew the meta-agent designed — key for evaluation
                "instructions": auto_raw.get("generated_config", {}),
            }
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return os.path.abspath(path)