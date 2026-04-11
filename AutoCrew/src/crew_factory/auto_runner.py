import os
import time
import json
from src.meta_agent.orchestrator import MetaAgentOrchestrator
from src.crew_factory.crew_builder import CrewBuilder
from src.hitl.human_gate import human_gate


def run_auto_pipeline(llm, dataset_path, task_desc, llm_provider_name="ollama"):
    results = {
        "mode": "auto",
        "dataset": dataset_path,
        "task": task_desc,
    }

    if not os.path.exists(dataset_path):
        print(f"\nERROR: Dataset not found: {dataset_path}")
        return None

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
        modeling_output = modeling_crew.kickoff()
    except Exception as e:
        print(f"\nModeling Crew Error: {e}")
        modeling_output = f"Modeling crew failed: {str(e)[:500]}"
    modeling_time = time.time() - model_start
    results["modeling_time_sec"] = round(modeling_time, 2)
    results["modeling_output"] = str(modeling_output)
    print(f"\nModeling completed in {modeling_time:.1f}s")
    print("\nModeling Output:")
    output_str = str(modeling_output)
    print(output_str[:3000] if len(output_str) > 3000 else output_str)

    print("\nWaiting 60s for API rate limit to reset...")
    time.sleep(60)

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
        # Inject modeling output into first task and dataset path into ALL tasks
        mrm_config["tasks"][0]["description"] = (
            mrm_config["tasks"][0].get("description", "")
            + f"\n\nModeling Output:\n{str(modeling_output)[:3000]}"
        )
        for task in mrm_config["tasks"]:
            if "description" in task:
                task["description"] = (
                    f"IMPORTANT: The real dataset is at '{dataset_path}'. "
                    f"NEVER generate synthetic or fake data. Always load from that path.\n\n"
                    + task["description"]
                )
        mrm_crew = builder.build_crew(mrm_config)

    try:
        final_verdict = mrm_crew.kickoff()
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
