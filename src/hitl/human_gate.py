import sys


def human_gate(modeling_output, phase="MRM Audit"):
    """
    Human-in-the-loop intervention point.
    Shows modeling output and asks whether to proceed.
    Used by BOTH manual and auto paths.
    
    Returns:
        True if human approves, False otherwise.
    """
    print("\n" + "=" * 50)
    print(f"HUMAN INTERVENTION ({phase})")
    print("=" * 50)

    print("\nModeling Output Summary:")
    # Show truncated output if too long
    output_str = str(modeling_output)
    if len(output_str) > 2000:
        print(output_str[:2000] + "\n... [truncated]")
    else:
        print(output_str)

    print("\n" + "-" * 50)
    proceed = input(f"Proceed to {phase}? (y/n): ").strip().lower()

    if proceed != "y":
        print("Human chose to stop. Exiting.")
        return False

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
