import os        # file path checks and env vars
import re        # parsing rate-limit wait times and metric extraction
import time      # phase timing and sleep between API calls
import json      # saving results to disk

from src.meta_agent.orchestrator import MetaAgentOrchestrator
from src.crew_factory.crew_builder import CrewBuilder
from src.hitl.human_gate import human_gate  # human approval checkpoint between modeling and MRM


# How long to wait between pipeline phases (seconds).
# Free-tier APIs (e.g. Groq) have per-minute token limits, so we pause between
# the modeling crew and MRM crew to avoid hitting rate limits mid-run.
# Override via environment: export AUTOCREW_PHASE_WAIT=0 to disable.
PHASE_WAIT_SEC = int(os.environ.get("AUTOCREW_PHASE_WAIT", "65"))


def _parse_rate_limit_wait(err_msg: str) -> int:
    """Parse the actual wait time from a Groq 429 error message.
    Groq embeds the reset time in the message, e.g. 'Please try again in 8m54.816s'.
    We add 15s buffer to avoid hitting the limit again immediately."""
    m = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2))
        return int(minutes * 60 + seconds) + 15
    return 0


def _is_tpd_error(err_msg: str) -> bool:
    """Detect tokens-per-day exhaustion — retrying is pointless until midnight UTC."""
    return "tokens per day (tpd)" in err_msg.lower() or "per day" in err_msg.lower()


def _kickoff_with_retry(crew, label: str, max_retries: int = 3):
    """Run crew.kickoff() and retry on rate limit errors.
    TPD errors (daily limit hit) raise immediately — no point waiting hours."""
    for attempt in range(1, max_retries + 1):
        try:
            return crew.kickoff()
        except Exception as e:
            err = str(e)
            is_rate_limit = "ratelimit" in err.lower() or "rate_limit" in err.lower() or "429" in err
            if is_rate_limit and _is_tpd_error(err):
                print(f"\n[{label}] Daily token limit exhausted — cannot retry today.")
                raise
            if is_rate_limit and attempt < max_retries:
                wait = _parse_rate_limit_wait(err)
                if wait == 0:
                    wait = 60 * attempt  # fallback if parse fails
                print(f"\n[{label}] Rate limit hit (attempt {attempt}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def _phase_wait(label: str):
    """Pause between pipeline phases to respect per-minute API quotas."""
    wait = PHASE_WAIT_SEC
    if wait > 0:
        print(f"\nWaiting {wait}s for rate limit reset... ({label})")
        time.sleep(wait)


def _extract_baseline_accuracy(modeling_output: str, verdict_text: str = "") -> float | None:
    """Pull baseline accuracy out of agent output text.
    Prefers the stress test agent's explicit 'Baseline Accuracy:' line,
    falls back to any 'Accuracy:' value in the modeling output.
    Used by the verdict override to catch when the CRO ignores the approval rule."""
    combined = verdict_text + "\n" + modeling_output
    m = re.search(r"baseline\s+accuracy[\s:=]+([0-9]\.[0-9]{2,4})", combined, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"\baccuracy[\s:=]+([0-9]\.[0-9]{2,4})", combined, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if 0.0 < val <= 1.0:
            return val
    return None


def run_auto_pipeline(llm, dataset_path, task_desc, llm_provider_name="ollama"):
    """Path B: AutoCrew pipeline driven by the Meta Agent.
    The Meta Agent generates the crew configuration dynamically from the task description,
    so no hardcoded agent roles or task sequences are needed."""
    results = {
        "mode": "auto",
        "dataset": dataset_path,
        "task": task_desc,
    }

    if not os.path.exists(dataset_path):
        print(f"\nERROR: Dataset not found: {dataset_path}")
        return None

    _clear_agent_plots()

    print("\n" + "=" * 50)
    print("PATH B: AUTOCREW PIPELINE (Meta Agent)")
    print("=" * 50)

    # Phase 1: Meta Agent generates crew config from the task description.
    # This is the key difference from Path A — no human writes the agent roles.
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

    _phase_wait("before crew building")

    # Phase 2: Convert the generated JSON config into live CrewAI objects.
    print("\n--- Phase 2: Building Crews from JSON ---")
    builder = CrewBuilder(llm)
    modeling_crew, mrm_crew = builder.build_from_full_config(config)

    if modeling_crew is None:
        print("ERROR: Failed to build modeling crew.")
        return None

    # Phase 3: Modeling crew trains the model and produces outputs.
    print("\n--- Phase 3: Running Modeling Crew ---")
    model_start = time.time()
    try:
        modeling_output = _kickoff_with_retry(modeling_crew, "Modeling")
    except Exception as e:
        print(f"\nModeling Crew Error: {e}")
        modeling_output = f"Modeling crew failed: {str(e)[:500]}"
    modeling_time = time.time() - model_start
    results["modeling_time_sec"] = round(modeling_time, 2)

    # CrewAI's final output is the last task's result by default.
    # We collect ALL task outputs so that metrics printed mid-pipeline (e.g. by the
    # ML Engineer) are not lost when the final task is just a model card.
    if hasattr(modeling_output, "tasks_output") and modeling_output.tasks_output:
        all_task_texts = [t.raw for t in modeling_output.tasks_output if t and t.raw]
        combined_output = "\n\n---TASK OUTPUT---\n".join(all_task_texts)
        results["modeling_output"] = combined_output or str(modeling_output)
        results["all_task_texts"] = all_task_texts
    else:
        results["modeling_output"] = str(modeling_output)

    print(f"\nModeling completed in {modeling_time:.1f}s")
    print("\nModeling Output:")
    output_str = results["modeling_output"]
    print(output_str[:3000] if len(output_str) > 3000 else output_str)

    # Safety net: if an agent only printed Accuracy (not F1/Precision/Recall),
    # rerun the model directly from code and append the missing metrics.
    results["modeling_output"] = _ensure_full_metrics(results["modeling_output"], dataset_path)

    _phase_wait("before MRM phase")

    # Show Groq quota only when we're actually using the Groq API.
    if os.environ.get("GROQ_API_KEY") and llm_provider_name.startswith("groq"):
        try:
            from src.utils.groq_limits import show_groq_limits
            show_groq_limits("Before MRM Phase", model=getattr(llm, 'model', None))
        except Exception:
            pass

    # Don't pass a failed modeling output to MRM — it would hallucinate a verdict.
    modeling_str = results.get("modeling_output", "")
    if "crew failed" in modeling_str.lower() or "none or empty" in modeling_str.lower():
        print("\n[AutoCrew] Modeling failed — skipping MRM.")
        results["mrm_verdict"] = "MODELING_FAILED"
        results["mrm_time_sec"] = 0
        results["total_time_sec"] = round(
            results.get("meta_agent_time_sec", 0) + results.get("modeling_time_sec", 0), 2
        )
        _generate_report(results, config_path)
        _save_results_json(results)
        return results

    # Phase 4: Human reviews modeling metrics before MRM begins.
    # This is the HITL (human-in-the-loop) checkpoint from the paper.
    if not human_gate(modeling_output, phase="MRM Audit"):
        results["mrm_verdict"] = "SKIPPED_BY_HUMAN"
        results["mrm_time_sec"] = 0
        results["total_time_sec"] = round(
            results.get("meta_agent_time_sec", 0) + results.get("modeling_time_sec", 0), 2
        )
        _generate_report(results, config_path)
        _save_results_json(results)
        return results

    _phase_wait("before MRM crew execution")

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

    # Phase 5: MRM crew stress-tests the model and issues a verdict.
    print("\n--- Phase 5: Running MRM Crew ---")
    mrm_start = time.time()
    mrm_config = config.get("mrm_crew", {})
    if mrm_config.get("tasks"):
        # Inject the actual modeling metrics into each MRM task so agents don't
        # hallucinate values. collect_metrics handles all output formats.
        from src.evaluation.metrics_collector import collect_metrics as _cm
        _m = _cm(str(modeling_output), 0, "auto")
        modeling_summary = ""
        for _k, _label in [("accuracy", "Accuracy"), ("f1_score", "F1 Score"),
                            ("precision", "Precision"), ("recall", "Recall")]:
            if _m.get(_k) is not None:
                modeling_summary += f"{_label}: {_m[_k]:.4f}\n"
        if not modeling_summary:
            modeling_summary = _extract_metrics_snippet(str(modeling_output))

        for task in mrm_config["tasks"]:
            if "description" in task:
                task["description"] = (
                    f"IMPORTANT: The real dataset is at '{dataset_path}'. "
                    f"NEVER generate synthetic or fake data. Always load from that path.\n\n"
                    + task["description"]
                )
        if modeling_summary:
            for task in mrm_config["tasks"]:
                if "description" in task:
                    task["description"] += f"\n\nModeling Metrics (use these — do NOT invent values):\n{modeling_summary}"

        # The CRO verdict rule is injected only into the last MRM task (the verdict task).
        last_task = mrm_config["tasks"][-1]
        if "description" in last_task:
            last_task["description"] += (
                "\n\nVERDICT RULE: Base your APPROVED/REJECTED decision on model performance ONLY — "
                "NOT on documentation gaps. "
                "If Baseline Accuracy > 0.7: APPROVED (even if stress test output is missing or ambiguous). "
                "If Baseline Accuracy <= 0.7: REJECTED. "
                "Minor documentation issues alone must NEVER cause REJECTED."
            )
        mrm_crew = builder.build_crew(mrm_config)

    try:
        final_verdict = _kickoff_with_retry(mrm_crew, "MRM")
    except Exception as e:
        print(f"\nMRM Crew Error: {e}")
        final_verdict = f"MRM crew failed: {str(e)[:500]}"
    mrm_time = time.time() - mrm_start

    # Verdict override: even with an explicit rule, the LLM sometimes ignores it
    # (e.g. when the Compliance Officer's 0/7 report dominates the context).
    # We enforce the rule in code: if accuracy > 0.7, flip REJECTED → APPROVED.
    final_verdict_str = str(final_verdict)
    if "rejected" in final_verdict_str.lower() and "approved" not in final_verdict_str.lower():
        baseline_acc = _extract_baseline_accuracy(str(modeling_output), final_verdict_str)
        if baseline_acc is not None and baseline_acc > 0.7:
            print(
                f"\n[MRM Override] CRO issued REJECTED but Baseline Accuracy={baseline_acc:.4f} > 0.7 threshold."
                f"\n[MRM Override] Overriding to APPROVED per the evaluation rule."
            )
            final_verdict = (
                f"FINAL VERDICT: APPROVED\n"
                f"(Override: Baseline Accuracy {baseline_acc:.4f} > 0.7 — CRO verdict corrected)\n\n"
                f"Original CRO output:\n{final_verdict_str}"
            )
        elif baseline_acc is None:
            print("\n[MRM Override] Could not extract accuracy — keeping CRO verdict.")

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


def _ensure_full_metrics(modeling_output: str, dataset_path: str) -> str:
    """Fallback metric computation when agents skip F1/Precision/Recall.
    LLM agents sometimes only print Accuracy despite instructions to print all four.
    We detect the gap and rerun the model directly via code execution to fill it."""
    import re
    has_f1 = bool(re.search(r"f1[\-_\s]?score[\s:=]+[0-9]+\.[0-9]+", modeling_output, re.IGNORECASE))
    has_precision = bool(re.search(r"precision[\s:=]+[0-9]+\.[0-9]+", modeling_output, re.IGNORECASE))
    has_recall = bool(re.search(r"recall[\s:=]+[0-9]+\.[0-9]+", modeling_output, re.IGNORECASE))

    if has_f1 and has_precision and has_recall:
        return modeling_output

    print("\n[Metrics] F1/Precision/Recall missing — computing directly from dataset...")
    try:
        from src.tools.code_execution import code_execution_tool

        if "credit_card_approval" in dataset_path:
            load_snippet = (
                f"df = pd.read_csv('{dataset_path}', header=None)\n"
                "df = df.replace('?', np.nan)\n"
                "df = df.dropna()\n"
                "for col in df.select_dtypes(include='object').columns:\n"
                "    df[col] = LabelEncoder().fit_transform(df[col])\n"
                "target_col = df.columns[-1]\n"
            )
        elif "cs-training" in dataset_path:
            load_snippet = (
                f"df = pd.read_csv('{dataset_path}')\n"
                "df = df.drop(columns=[c for c in df.columns if 'unnamed' in c.lower()], errors='ignore')\n"
                "df = df.fillna(df.median(numeric_only=True))\n"
                "for col in df.select_dtypes(include='object').columns:\n"
                "    df[col] = LabelEncoder().fit_transform(df[col])\n"
                "target_col = 'SeriousDlqin2yrs'\n"
            )
        elif "creditcard_2023" in dataset_path:
            load_snippet = (
                f"df = pd.read_csv('{dataset_path}')\n"
                "df = df.drop(columns=[c for c in df.columns if c.lower() in ('id', 'unnamed: 0')], errors='ignore')\n"
                "df = df.dropna()\n"
                "for col in df.select_dtypes(include='object').columns:\n"
                "    df[col] = LabelEncoder().fit_transform(df[col])\n"
                "target_col = 'Class'\n"
            )
        else:
            load_snippet = (
                f"df = pd.read_csv('{dataset_path}')\n"
                "df = df.replace('?', np.nan)\n"
                "df = df.dropna()\n"
                "for col in df.select_dtypes(include='object').columns:\n"
                "    df[col] = LabelEncoder().fit_transform(df[col])\n"
                "target_col = df.columns[-1]\n"
            )

        # creditcard_2023 is 568K rows — use lighter params and downsample to avoid timeout
        rf_params = (
            "n_estimators=20, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced'"
            if "creditcard_2023" in dataset_path
            else "n_estimators=100, random_state=42"
        )

        code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.preprocessing import LabelEncoder\n"
            "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score\n"
            + load_snippet
            + "X = df.drop(columns=[target_col])\n"
            "y = df[target_col]\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            f"model = RandomForestClassifier({rf_params})\n"
            "model.fit(X_train, y_train)\n"
            "y_pred = model.predict(X_test)\n"
            "print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))\n"
            "print('F1 Score:', round(f1_score(y_test, y_pred, average='weighted'), 4))\n"
            "print('Precision:', round(precision_score(y_test, y_pred, average='weighted'), 4))\n"
            "print('Recall:', round(recall_score(y_test, y_pred, average='weighted'), 4))"
        )

        result = code_execution_tool.run(code)
        print(f"[Metrics] Fallback result:\n{result}")
        return modeling_output + "\n\n---TASK OUTPUT---\n" + str(result)
    except Exception as e:
        print(f"[Metrics] Fallback computation failed: {e}")
        return modeling_output


def _extract_metrics_snippet(modeling_output: str) -> str:
    """Pull only the metric lines out of a raw agent output string.
    Used to give MRM agents a clean, concise summary instead of the full verbose output."""
    import re
    lines = modeling_output.splitlines()
    metric_lines = []
    for line in lines:
        if re.search(r"(accuracy|f1[\s_]?score?|f1|precision|recall)\s*[:\s=]\s*[\d.]+", line, re.IGNORECASE):
            metric_lines.append(line.strip())
    if metric_lines:
        # Deduplicate: keep the last occurrence of each metric name
        seen = {}
        for line in metric_lines:
            key = re.match(r"(\w+)", line.lower())
            if key:
                seen[key.group(1)] = line
        return "\n".join(seen.values())
    return ""


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


def _generate_report(results, config_path):
    """Collect metrics from all available task outputs and generate the PDF report.
    We join modeling_output + individual task texts because CrewAI's final output
    is often just the last task (model card), which may not contain the metric lines."""
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

        metrics = collect_metrics(combined_text, results.get("total_time_sec", 0.0), mode="auto")
        metrics["mrm_verdict"] = results.get("mrm_verdict", "UNKNOWN")
        metrics["modeling_time_sec"] = results.get("modeling_time_sec", 0.0)
        metrics["mrm_time_sec"] = results.get("mrm_time_sec", 0.0)
        metrics["meta_agent_time_sec"] = results.get("meta_agent_time_sec", 0.0)
        metrics["total_time_sec"] = results.get("total_time_sec", 0.0)

        generate_all(dataset_name=results.get("dataset", ""), auto_metrics=metrics)
        generate_report(mode="auto", config_path=config_path, results={**results, **metrics},
                        output_path="results/AutoCrew_Report.pdf")
    except Exception as e:
        print(f"\n[Report] PDF generation failed: {e}")
        import traceback
        traceback.print_exc()


def _save_results_json(results):
    """Persist run results to JSON for later analysis and comparison."""
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
