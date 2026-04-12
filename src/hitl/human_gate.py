import sys
import time

def human_gate(modeling_output, phase="MRM Audit"):
    """
    Human-in-the-loop intervention point.
    Shows a clean summary and asks whether to proceed.
    Used by BOTH manual and auto paths.
    
    Returns:
        True if human approves, False otherwise.
    """
    output_str = str(modeling_output)

    # Extract key metrics if present — handles plain text, markdown bold, and colon formats
    import re
    metrics_found = []
    seen_labels = set()
    text_lower = output_str.lower()
    for metric in ["accuracy", "f1 score", "f1", "precision", "recall"]:
        label = "F1 Score" if metric in ("f1 score", "f1") else metric.title()
        if label in seen_labels:
            continue  # already captured this metric under a different alias
        for pattern in [
            rf"\*\*{re.escape(metric)}\*\*\s*[:=]?\s*([\d.]+)",   # **Accuracy**: 0.93
            rf"{re.escape(metric)}\s*[:=]\s*([\d.]+)",             # Accuracy: 0.93
            rf"{re.escape(metric)}.*?(0\.\d{{2,4}})",              # Accuracy ... 0.9317
        ]:
            match = re.search(pattern, text_lower)
            if match:
                val = float(match.group(1))
                if 0 < val <= 1:
                    metrics_found.append(f"  {label}: {val:.4f}")
                    seen_labels.add(label)
                    break

    print("\n")
    print("=" * 55)
    print(f"  HUMAN REVIEW REQUIRED — {phase}")
    print("=" * 55)

    if metrics_found:
        print("\n  Metrics extracted from modeling output:")
        for m in metrics_found:
            print(m)
    else:
        print("\n  (No metrics could be extracted automatically)")
        print(f"  Output length: {len(output_str)} characters")

    print("\n" + "-" * 55)
    print(f"  Type 'y' to proceed to {phase}")
    print(f"  Type 'n' to stop the pipeline")
    print("-" * 55)

    
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(2)

    while True:
        proceed = input("\n  >>> Your choice (y/n): ").strip().lower()
        if proceed in ["y", "n"]:
            break
        print("  Please enter 'y' or 'n'.")

    if proceed != "y":
        print("\n  Pipeline stopped by user.\n")
        return False

    print(f"\n  Proceeding to {phase}...\n")
    return True


def human_feedback_prompt(agent_name: str) -> str:
    """
    Optional: ask human for additional instructions for an agent.
    Returns the feedback string or empty string if none.
    """
    print(f"\n[HITL] Agent '{agent_name}' completed its task.")
    feedback = input("Additional instructions (or 'skip' to continue): ").strip()

    if feedback.lower() in ["skip", "end", ""]:
        return ""

    return feedback
