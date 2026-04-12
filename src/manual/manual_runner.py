import os
import re
import time
from src.manual.modeling_crew import ManualModelingCrew
from src.manual.mrm_crew import ManualMRMCrew
from src.hitl.human_gate import human_gate


def _parse_rate_limit_wait(err_msg: str) -> int:
    """Extract required wait seconds from a Groq rate limit error. Returns 0 if not found."""
    m = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2))
        return int(minutes * 60 + seconds) + 15
    return 0


def _is_tpd_error(err_msg: str) -> bool:
    """Return True if the error is a tokens-per-day exhaustion (unrecoverable today)."""
    return "tokens per day (tpd)" in err_msg.lower() or "per day" in err_msg.lower()


def _run_with_retry(fn, label: str, max_retries: int = 3):
    """Call fn() with smart backoff on rate-limit errors.
    Parses the actual reset time from the error instead of using fixed waits.
    TPD (tokens-per-day) errors are NOT retried — retrying burns tokens and always fails."""
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
                print(f"\n[{label}] Daily token limit (TPD) exhausted — retrying would fail. Raising immediately.")
                raise
            if is_rate_limit and attempt < max_retries:
                wait = _parse_rate_limit_wait(err)
                if wait == 0:
                    wait = 60 * attempt
                print(f"\n[{label}] Rate limit hit (attempt {attempt}/{max_retries}). "
                      f"Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise


def run_manual_pipeline(llm, dataset_path, task_desc):
    """
    PATH A: Run the base paper's hardcoded pipeline.
    Returns a dict with results and timing for comparison.
    """
    results = {
        "mode": "manual",
        "dataset": dataset_path,
        "task": task_desc,
    }

    if not os.path.exists(dataset_path):
        print(f"\nERROR: Dataset not found: {dataset_path}")
        return None

    # Clear agent plots from any previous run
    _clear_agent_plots()

    # --- Modeling Phase ---
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

    # Collect ALL task outputs so metrics from the training task are not lost
    # if str(modeling_output) only returns the last task (e.g. model card).
    if hasattr(modeling_output, "tasks_output") and modeling_output.tasks_output:
        all_task_texts = [
            t.raw for t in modeling_output.tasks_output if t and t.raw
        ]
        combined_output = "\n\n---TASK OUTPUT---\n".join(all_task_texts)
        results["modeling_output"] = combined_output or str(modeling_output)
    else:
        results["modeling_output"] = str(modeling_output)

    print(f"\nModeling completed in {modeling_time:.1f}s")
    print("\nModeling Output:")
    print(results["modeling_output"])

    # Show remaining Groq quota before MRM phase
    if os.environ.get("GROQ_API_KEY"):
        try:
            from src.utils.groq_limits import show_groq_limits
            groq_model = getattr(llm, 'model', None)
            show_groq_limits("Before MRM Phase", model=groq_model)
        except Exception:
            pass

    # Guard: if modeling completely failed, do not pass garbage to MRM
    modeling_str = results.get("modeling_output", "")
    if "crew failed" in modeling_str.lower() or "none or empty" in modeling_str.lower():
        print("\n[Manual] Modeling crew failed — skipping MRM to prevent hallucinated verdict.")
        results["mrm_verdict"] = "MODELING_FAILED"
        return results

    # --- Human Gate ---
    if not human_gate(modeling_output, phase="MRM Audit"):
        results["mrm_verdict"] = "SKIPPED_BY_HUMAN"
        return results

    # --- MRM Phase ---
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
    """Remove plots from previous run so the report only shows current run's graphs."""
    plots_dir = os.path.join("figures", "agent_plots")
    if os.path.exists(plots_dir):
        for f in os.listdir(plots_dir):
            if f.endswith(".png"):
                try:
                    os.remove(os.path.join(plots_dir, f))
                except OSError:
                    pass


def _generate_report(results):
    try:
        from src.evaluation.report_generator import generate_report
        generate_report(mode="manual", results=results, output_path="results/Manual_Report.pdf")
    except Exception as e:
        print(f"\n[Report] PDF generation failed: {e}")


def _save_results_json(results):
    import json
    os.makedirs("results", exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/manual_results_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Results] Saved to: {filepath}")
