import os
import sys
import time
from src.manual.modeling_crew import ManualModelingCrew
from src.manual.mrm_crew import ManualMRMCrew
from src.hitl.human_gate import human_gate


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

    # --- Modeling Phase ---
    print("\n" + "=" * 50)
    print("PATH A: MANUAL PIPELINE (Base Paper)")
    print("=" * 50)

    print(f"\nStarting Modeling Phase for: {task_desc}")
    start_time = time.time()

    model_crew = ManualModelingCrew(llm)
    modeling_output = model_crew.run(dataset_path, task_desc)

    modeling_time = time.time() - start_time
    results["modeling_time_sec"] = round(modeling_time, 2)
    results["modeling_output"] = str(modeling_output)

    print(f"\nModeling completed in {modeling_time:.1f}s")
    print("\nModeling Output:")
    print(modeling_output)

    # --- Human Gate ---
    if not human_gate(modeling_output, phase="MRM Audit"):
        results["mrm_verdict"] = "SKIPPED_BY_HUMAN"
        return results

    # --- MRM Phase ---
    print("\nStarting MRM Audit Phase...")
    start_time = time.time()

    mrm_crew = ManualMRMCrew(llm)
    final_verdict = mrm_crew.run(modeling_output)

    mrm_time = time.time() - start_time
    results["mrm_time_sec"] = round(mrm_time, 2)
    results["mrm_verdict"] = str(final_verdict)

    print(f"\nMRM completed in {mrm_time:.1f}s")
    print("\n" + "=" * 50)
    print(f"FINAL VERDICT:\n{final_verdict}")

    results["total_time_sec"] = round(
        results["modeling_time_sec"] + results["mrm_time_sec"], 2
    )

    return results
