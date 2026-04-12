import os
import re
import time
import json
from src.meta_agent.orchestrator import MetaAgentOrchestrator
from src.crew_factory.crew_builder import CrewBuilder
from src.hitl.human_gate import human_gate


def _parse_rate_limit_wait(err_msg: str) -> int:
    """
    Extract the required wait time in seconds from a Groq rate limit error message.
    Handles: 'Please try again in 8m54.816s' or 'Please try again in 54.3s'
    Returns parsed seconds + 15s buffer, or 0 if not found.
    """
    m = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2))
        return int(minutes * 60 + seconds) + 15  # buffer
    return 0


def _is_tpd_error(err_msg: str) -> bool:
    """Return True if the error is a tokens-per-day exhaustion (unrecoverable today)."""
    return "tokens per day (tpd)" in err_msg.lower() or "per day" in err_msg.lower()


def _kickoff_with_retry(crew, label: str, max_retries: int = 3):
    """Run crew.kickoff() with smart backoff on rate-limit errors.
    Parses the actual reset time from the error message instead of using fixed waits.
    TPD (tokens-per-day) errors are NOT retried — retrying burns tokens and always fails."""
    for attempt in range(1, max_retries + 1):
        try:
            return crew.kickoff()
        except Exception as e:
            err = str(e)
            is_rate_limit = "ratelimit" in err.lower() or "rate_limit" in err.lower() or "429" in err
            if is_rate_limit and _is_tpd_error(err):
                print(f"\n[{label}] Daily token limit (TPD) exhausted — retrying would fail. Raising immediately.")
                raise
            if is_rate_limit and attempt < max_retries:
                wait = _parse_rate_limit_wait(err)
                if wait == 0:
                    wait = 60 * attempt  # fallback: 60s, 120s
                print(f"\n[{label}] Rate limit hit (attempt {attempt}/{max_retries}). "
                      f"Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise


def run_auto_pipeline(llm, dataset_path, task_desc, llm_provider_name="ollama"):
    results = {
        "mode": "auto",
        "dataset": dataset_path,
        "task": task_desc,
    }

    if not os.path.exists(dataset_path):
        print(f"\nERROR: Dataset not found: {dataset_path}")
        return None

    # Clear agent plots from any previous run
    _clear_agent_plots()

    print("\n" + "=" * 50)
    print("PATH B: AUTOCREW PIPELINE (Meta Agent)")
    print("=" * 50)

    print("\n--- Phase 1: Meta Agent Config Generation ---")
    meta_start = time.time()
    orchestrator = MetaAgentOrchestrator(llm, llm_provider_name)
    config = orchestrator.run(task_desc, dataset_path)
    meta_time = time.time() - meta_start
    results["meta_agent_time_sec"] = round(meta_time, 2)
    print(f"\nMeta Agent completed in {meta_time:.1f}s")

    config_dir = "configs/generated"
    config_files = sorted([f for f in os.listdir(config_dir) if f.endswith(".json")]) if os.path.exists(config_dir) else []
    config_path = os.path.join(config_dir, config_files[-1]) if config_files else None
    results["config_path"] = config_path
    results["generated_config"] = config

    print("\nWaiting 60s for API rate limit to reset...")
    time.sleep(60)

    print("\n--- Phase 2: Building Crews from JSON ---")
    builder = CrewBuilder(llm)
    modeling_crew, mrm_crew = builder.build_from_full_config(config)

    if modeling_crew is None:
        print("ERROR: Failed to build modeling crew.")
        return None

    print("\n--- Phase 3: Running Modeling Crew ---")
    model_start = time.time()
    try:
        modeling_output = _kickoff_with_retry(modeling_crew, "Modeling")
    except Exception as e:
        print(f"\nModeling Crew Error: {e}")
        modeling_output = f"Modeling crew failed: {str(e)[:500]}"
    modeling_time = time.time() - model_start
    results["modeling_time_sec"] = round(modeling_time, 2)

    # Collect ALL task outputs so metrics from intermediate tasks (e.g. training)
    # are not lost when the last task is a model card or documentation task.
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
    output_str = results["modeling_output"]
    print(output_str[:3000] if len(output_str) > 3000 else output_str)

    print("\nWaiting 60s for API rate limit to reset...")
    time.sleep(60)

    # Show remaining Groq quota before MRM phase (works for all Groq models)
    if os.environ.get("GROQ_API_KEY") and llm_provider_name.startswith("groq"):
        try:
            from src.utils.groq_limits import show_groq_limits
            groq_model = getattr(llm, 'model', None)
            show_groq_limits("Before MRM Phase", model=groq_model)
        except Exception:
            pass

    # Guard: if modeling completely failed, do not pass garbage to MRM
    modeling_str = results.get("modeling_output", "")
    if "crew failed" in modeling_str.lower() or "none or empty" in modeling_str.lower():
        print("\n[AutoCrew] Modeling crew failed — skipping MRM to prevent hallucinated verdict.")
        results["mrm_verdict"] = "MODELING_FAILED"
        results["mrm_time_sec"] = 0
        results["total_time_sec"] = round(
            results.get("meta_agent_time_sec", 0) + results.get("modeling_time_sec", 0), 2
        )
        _generate_report(results, config_path)
        _save_results_json(results)
        return results

    if not human_gate(modeling_output, phase="MRM Audit"):
        results["mrm_verdict"] = "SKIPPED_BY_HUMAN"
        results["mrm_time_sec"] = 0
        results["total_time_sec"] = round(
            results.get("meta_agent_time_sec", 0) + results.get("modeling_time_sec", 0), 2
        )
        _generate_report(results, config_path)
        _save_results_json(results)
        return results

    print("\nWaiting 60s for API rate limit to reset...")
    time.sleep(60)

    if mrm_crew is None:
        print("WARNING: No MRM crew was generated.")
        results["mrm_verdict"] = "NO_MRM_CREW"
        results["mrm_time_sec"] = 0
        results["total_time_sec"] = round(
            results.get("meta_agent_time_sec", 0) + results.get("modeling_time_sec", 0), 2
        )
        _generate_report(results, config_path)
        _save_results_json(results)
        return results

    print("\n--- Phase 5: Running MRM Crew ---")
    mrm_start = time.time()
    mrm_config = config.get("mrm_crew", {})
    if mrm_config.get("tasks"):
        # Extract only the metric lines from modeling output (keep context short).
        # Full modeling output can include large EDA reports that bloat MRM task descriptions.
        modeling_summary = _extract_metrics_snippet(str(modeling_output))
        for task in mrm_config["tasks"]:
            if "description" in task:
                task["description"] = (
                    f"IMPORTANT: The real dataset is at '{dataset_path}'. "
                    f"NEVER generate synthetic or fake data. Always load from that path.\n\n"
                    + task["description"]
                )
        # Inject modeling metrics into ALL MRM tasks so every agent (including the verdict agent)
        # has access to the real Accuracy/F1/Precision/Recall values. Without this, the CRO
        # only sees "Stress Test Passed" from the prior task and hallucinates rounded metrics.
        if modeling_summary:
            for task in mrm_config["tasks"]:
                if "description" in task:
                    task["description"] = (
                        task["description"]
                        + f"\n\nModeling Metrics (use these — do NOT invent values):\n{modeling_summary}"
                    )
        # Append verdict fairness note to the LAST task (typically the verdict/approval task)
        last_task = mrm_config["tasks"][-1]
        if "description" in last_task:
            last_task["description"] = (
                last_task["description"]
                + "\n\nVERDICT RULE: Base your APPROVED/REJECTED decision on model performance "
                "(Accuracy and Stress Test) — NOT on documentation gaps. "
                "If Accuracy > 0.7 and Stress Test Passed: APPROVED. "
                "Minor documentation issues alone must NOT cause REJECTED."
            )
        mrm_crew = builder.build_crew(mrm_config)

    try:
        final_verdict = _kickoff_with_retry(mrm_crew, "MRM")
    except Exception as e:
        print(f"\nMRM Crew Error: {e}")
        final_verdict = f"MRM crew failed: {str(e)[:500]}"
    mrm_time = time.time() - mrm_start
    results["mrm_time_sec"] = round(mrm_time, 2)
    results["mrm_verdict"] = str(final_verdict)
    print(f"\nMRM completed in {mrm_time:.1f}s")
    print("\n" + "=" * 50)
    print(f"FINAL VERDICT:\n{final_verdict}")

    results["total_time_sec"] = round(
        results["meta_agent_time_sec"] + results["modeling_time_sec"] + results["mrm_time_sec"], 2
    )
    results["generated_config"] = config
    _generate_report(results, config_path)
    _save_results_json(results)
    return results


def _extract_metrics_snippet(modeling_output: str) -> str:
    """Extract only the metric lines from the modeling output to keep MRM task descriptions short.
    Looks for lines containing Accuracy, F1, Precision, Recall values. Returns empty string
    if no metrics are found (e.g. when modeling crew failed entirely)."""
    import re
    lines = modeling_output.splitlines()
    metric_lines = []
    for line in lines:
        if re.search(r"(accuracy|f1[\s_]?score?|f1|precision|recall)\s*[:\s=]\s*[\d.]+", line, re.IGNORECASE):
            metric_lines.append(line.strip())
    if metric_lines:
        # Deduplicate and take last occurrence of each metric (final result)
        seen = {}
        for line in metric_lines:
            key = re.match(r"(\w+)", line.lower())
            if key:
                seen[key.group(1)] = line
        return "\n".join(seen.values())
    return ""


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


def _generate_report(results, config_path):
    try:
        from src.evaluation.report_generator import generate_report
        generate_report(mode="auto", config_path=config_path, results=results, output_path="results/AutoCrew_Report.pdf")
    except Exception as e:
        print(f"\n[Report] PDF generation failed: {e}")
        print("[Report] Install reportlab: pip install reportlab")


def _save_results_json(results):
    os.makedirs("results", exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/auto_results_{timestamp}.json"
    save_data = {k: v for k, v in results.items() if k != "generated_config"}
    save_data["config_summary"] = {
        "modeling_agents": len(results.get("generated_config", {}).get("modeling_crew", {}).get("agents", [])),
        "mrm_agents": len(results.get("generated_config", {}).get("mrm_crew", {}).get("agents", [])),
    }
    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"[Results] Saved to: {filepath}")
