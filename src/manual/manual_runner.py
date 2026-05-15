import os        # file checks and env vars
import re        # rate-limit wait time parsing
import time      # phase timing and sleep
import json      # saving results to disk

from src.manual.modeling_crew import ManualModelingCrew
from src.manual.mrm_crew import ManualMRMCrew
from src.hitl.human_gate import human_gate  # human approval checkpoint between modeling and MRM


def _parse_rate_limit_wait(err_msg: str) -> int:
    """Parse the actual reset time from a Groq 429 error message.
    Returns the wait time in seconds plus a 15s buffer, or 0 if unparseable."""
    m = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2))
        return int(minutes * 60 + seconds) + 15
    return 0


def _is_tpd_error(err_msg: str) -> bool:
    """Detect tokens-per-day exhaustion. Retrying TPD errors is pointless — they don't reset until midnight UTC."""
    return "tokens per day (tpd)" in err_msg.lower() or "per day" in err_msg.lower()


def _run_with_retry(fn, label: str, max_retries: int = 3):
    """Call fn() with backoff on rate-limit errors.
    Parses the actual reset time from the Groq error instead of guessing with a fixed wait."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            err = str(e)
            is_rate_limit = (
                "ratelimit" in err.lower()
                or "rate_limit" in err.lower()
                or "429" in err
            )
            if is_rate_limit and _is_tpd_error(err):
                print(f"\n[{label}] Daily token limit exhausted — cannot retry today.")
                raise
            if is_rate_limit and attempt < max_retries:
                wait = _parse_rate_limit_wait(err)
                if wait == 0:
                    wait = 60
                print(f"\n[{label}] Rate limit hit (attempt {attempt}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def run_manual_pipeline(llm, dataset_path, task_desc):
    """Path A: The fixed manual pipeline from the base paper.
    Agent roles and task sequences are hardcoded — no meta-agent involved.
    This is the baseline we compare AutoCrew against."""
    results = {
        "mode": "manual",
        "dataset": dataset_path,
        "task": task_desc,
    }

    if not os.path.exists(dataset_path):
        print(f"\nERROR: Dataset not found: {dataset_path}")
        return None

    _clear_agent_plots()

    print("\n" + "=" * 50)
    print("PATH A: MANUAL PIPELINE (Base Paper)")
    print("=" * 50)

    print(f"\nStarting Modeling Phase for: {task_desc}")
    start_time = time.time()

    model_crew = ManualModelingCrew(llm)
    try:
        modeling_output = _run_with_retry(
            lambda: model_crew.run(dataset_path, task_desc),
            label="Modeling"
        )
    except Exception as e:
        print(f"\nModeling Crew Error: {e}")
        modeling_output = f"Modeling crew failed: {str(e)[:500]}"

    modeling_time = time.time() - start_time
    results["modeling_time_sec"] = round(modeling_time, 2)

    # CrewAI returns only the last task's output by default.
    # Collecting all task outputs ensures metrics printed by the ML Engineer
    # are not lost when the final task is just a model card write-up.
    if hasattr(modeling_output, "tasks_output") and modeling_output.tasks_output:
        all_task_texts = [t.raw for t in modeling_output.tasks_output if t and t.raw]
        combined_output = "\n\n---TASK OUTPUT---\n".join(all_task_texts)
        results["modeling_output"] = combined_output or str(modeling_output)
        results["all_task_texts"] = all_task_texts
    else:
        results["modeling_output"] = str(modeling_output)

    print(f"\nModeling completed in {modeling_time:.1f}s")
    print("\nModeling Output:")
    print(results["modeling_output"])

    # Show Groq quota only when we're actually using the Groq API.
    _provider = getattr(llm, 'model', '') or ''
    if os.environ.get("GROQ_API_KEY") and 'groq' in _provider.lower():
        try:
            from src.utils.groq_limits import show_groq_limits
            show_groq_limits("Before MRM Phase", model=_provider)
        except Exception:
            pass

    # Don't pass failed modeling output to MRM — it would hallucinate a verdict.
    modeling_str = results.get("modeling_output", "")
    if "crew failed" in modeling_str.lower() or "none or empty" in modeling_str.lower():
        print("\n[Manual] Modeling failed — skipping MRM.")
        results["mrm_verdict"] = "MODELING_FAILED"
        return results

    # HITL checkpoint: human reviews modeling metrics before MRM starts.
    if not human_gate(modeling_output, phase="MRM Audit"):
        results["mrm_verdict"] = "SKIPPED_BY_HUMAN"
        return results

    print("\nStarting MRM Audit Phase...")
    start_time = time.time()

    mrm_crew = ManualMRMCrew(llm)
    try:
        final_verdict = _run_with_retry(
            lambda: mrm_crew.run(modeling_output, dataset_path=dataset_path),
            label="MRM"
        )
    except Exception as e:
        print(f"\nMRM Crew Error: {e}")
        final_verdict = f"MRM crew failed: {str(e)[:500]}"

    mrm_time = time.time() - start_time
    results["mrm_time_sec"] = round(mrm_time, 2)
    results["mrm_verdict"] = str(final_verdict)

    print(f"\nMRM completed in {mrm_time:.1f}s")
    print("\n" + "=" * 50)
    print(f"FINAL VERDICT:\n{final_verdict}")

    results["total_time_sec"] = round(
        results["modeling_time_sec"] + results["mrm_time_sec"], 2
    )

    _generate_report(results)
    _save_results_json(results)
    return results


def _clear_agent_plots():
    """Remove PNG plots from the previous run so they don't appear in the current report."""
    plots_dir = os.path.join("figures", "agent_plots")
    if os.path.exists(plots_dir):
        for f in os.listdir(plots_dir):
            if f.endswith(".png"):
                try:
                    os.remove(os.path.join(plots_dir, f))
                except OSError:
                    pass


def _generate_report(results):
    """Collect metrics across all task outputs and write the PDF report.
    We join modeling_output + individual task texts because the final CrewAI output
    is often just the model card, which may not contain the numeric metric lines."""
    try:
        from src.evaluation.metrics_collector import collect_metrics
        from src.evaluation.generate_charts import generate_all
        from src.evaluation.report_generator import generate_report

        raw_parts = []
        if results.get("modeling_output"):
            raw_parts.append(str(results["modeling_output"]))
        for key in ("task_outputs", "all_task_texts"):
            if results.get(key):
                if isinstance(results[key], list):
                    raw_parts.extend([str(t) for t in results[key]])
                else:
                    raw_parts.append(str(results[key]))
        combined_text = "\n\n---TASK OUTPUT---\n".join(raw_parts) if raw_parts else ""

        metrics = collect_metrics(combined_text, results.get("total_time_sec", 0.0), mode="manual")
        metrics["mrm_verdict"] = results.get("mrm_verdict", "UNKNOWN")
        metrics["modeling_time_sec"] = results.get("modeling_time_sec", 0.0)
        metrics["mrm_time_sec"] = results.get("mrm_time_sec", 0.0)
        metrics["meta_agent_time_sec"] = 0.0  # manual path has no meta agent
        metrics["total_time_sec"] = results.get("total_time_sec", 0.0)

        generate_all(dataset_name=results.get("dataset", ""), manual_metrics=metrics)
        generate_report(mode="manual", results={**results, **metrics},
                        output_path="results/Manual_Report.pdf")
    except Exception as e:
        print(f"\n[Report] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()


def _save_results_json(results):
    """Persist run results to JSON for later analysis and comparison."""
    import json
    os.makedirs("results", exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/manual_results_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Results] Saved to: {filepath}")
