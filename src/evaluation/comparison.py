import os
import json
from datetime import datetime
from src.evaluation.metrics_collector import MetricsCollector


class ComparisonRunner:
    """
    Runs both manual and auto pipelines on the same dataset,
    collects metrics, and generates a comparison report.
    """

    def __init__(self):
        self.collector = MetricsCollector()

    def run_comparison(self, llm, dataset_path, task_desc, llm_choice_name="ollama"):
        """
        Run both paths and return comparison data.
        """
        print("\n" + "#" * 60)
        print("  COMPARISON MODE: Manual vs AutoCrew")
        print("#" * 60)

        results = {}

        # --- Run Manual (Path A) ---
        print("\n\n>>> Running PATH A: Manual (Base Paper)...")
        from src.manual.manual_runner import run_manual_pipeline
        manual_results = run_manual_pipeline(llm, dataset_path, task_desc)
        if manual_results:
            results["manual"] = self.collector.extract_metrics(manual_results)
        else:
            results["manual"] = {"error": "Manual pipeline failed"}

        # --- Run Auto (Path B) ---
        print("\n\n>>> Running PATH B: AutoCrew (Meta Agent)...")
        from src.crew_factory.auto_runner import run_auto_pipeline
        auto_results = run_auto_pipeline(llm, dataset_path, task_desc, llm_choice_name)
        if auto_results:
            results["auto"] = self.collector.extract_metrics(auto_results)
        else:
            results["auto"] = {"error": "Auto pipeline failed"}

        # Generate report
        report = self.generate_report(results)
        self._save_report(report, results)

        return results

    def generate_report(self, results: dict) -> str:
        """Generate a markdown comparison report."""
        manual = results.get("manual", {})
        auto = results.get("auto", {})

        report = []
        report.append("# AutoCrew vs Manual Pipeline — Comparison Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nDataset: {manual.get('dataset', auto.get('dataset', 'N/A'))}")
        report.append(f"Task: {manual.get('task', auto.get('task', 'N/A'))}")

        report.append("\n## Timing Comparison\n")
        report.append("| Metric | Manual | AutoCrew |")
        report.append("|--------|--------|----------|")
        report.append(f"| Meta Agent Setup | N/A | {auto.get('meta_agent_time_sec', 'N/A')}s |")
        report.append(f"| Modeling Time | {manual.get('modeling_time_sec', 'N/A')}s | {auto.get('modeling_time_sec', 'N/A')}s |")
        report.append(f"| MRM Time | {manual.get('mrm_time_sec', 'N/A')}s | {auto.get('mrm_time_sec', 'N/A')}s |")
        report.append(f"| Total Time | {manual.get('total_time_sec', 'N/A')}s | {auto.get('total_time_sec', 'N/A')}s |")

        report.append("\n## Model Performance\n")
        report.append("| Metric | Manual | AutoCrew |")
        report.append("|--------|--------|----------|")
        for m in ["accuracy", "f1_score", "precision", "recall"]:
            mv = manual.get(m, -1)
            av = auto.get(m, -1)
            mv_str = f"{mv:.4f}" if mv >= 0 else "N/A"
            av_str = f"{av:.4f}" if av >= 0 else "N/A"
            report.append(f"| {m.replace('_', ' ').title()} | {mv_str} | {av_str} |")

        report.append("\n## Crew Structure\n")
        report.append("| Metric | Manual | AutoCrew |")
        report.append("|--------|--------|----------|")
        report.append(f"| Modeling Agents | {manual.get('auto_modeling_agents', 'N/A')} | {auto.get('auto_modeling_agents', 'N/A')} |")
        report.append(f"| Modeling Tasks | {manual.get('auto_modeling_tasks', 'N/A')} | {auto.get('auto_modeling_tasks', 'N/A')} |")
        report.append(f"| MRM Agents | {manual.get('auto_mrm_agents', 'N/A')} | {auto.get('auto_mrm_agents', 'N/A')} |")
        report.append(f"| MRM Tasks | {manual.get('auto_mrm_tasks', 'N/A')} | {auto.get('auto_mrm_tasks', 'N/A')} |")

        report.append("\n## MRM Verdict\n")
        report.append(f"- Manual: {manual.get('mrm_verdict', 'N/A')}")
        report.append(f"- AutoCrew: {auto.get('mrm_verdict', 'N/A')}")

        report.append("\n## Key Observations\n")
        report.append("- Manual path requires hardcoding agents/tasks in Python for each new use case")
        report.append("- AutoCrew path dynamically generates the crew structure from a natural language prompt")
        report.append("- Both paths share the same tools, HITL gate, and CrewAI execution engine")

        return "\n".join(report)

    def _save_report(self, report: str, results: dict):
        """Save the comparison report and raw data."""
        os.makedirs("results", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save markdown report
        report_path = f"results/comparison_{timestamp}.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved: {report_path}")

        # Save raw data
        data_path = f"results/comparison_{timestamp}.json"
        with open(data_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Raw data saved: {data_path}")

        # Print report to console
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)

        # Regenerate charts with correct dataset
        try:
            from src.evaluation.generate_charts import generate_all
            ds = results.get("manual", {}).get("dataset", results.get("auto", {}).get("dataset", ""))
            generate_all(dataset_name=ds)
        except Exception as e:
            print(f"[Comparison] Chart regeneration failed: {e}")
            
        self.collector.save_records()
