import sys
import os
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
from src.config import get_llm, select_llm_interactive
from src.evaluation.terminal_logger import TerminalLogger


# Dataset options
DATASETS = {
    "1": {
        "path": "data/credit_card_approval.csv",
        "desc": (
            "Predict credit approval (Binary Classification). "
            "NOTE: This CSV has NO header row — use pd.read_csv(..., header=None). "
            "The target is the last column (column 15). "
            "Use header=None so column names become integers 0-15. "
            "Replace '?' with NaN, drop NaN rows, LabelEncoder on object columns."
        ),
        "has_header": False,
    },
    "2": {
        "path": "data/creditcard_2023.csv",
        "desc": (
            "Detect Fraud (Imbalanced Binary Classification). "
            "Target column: 'Class' (1=fraud, 0=legitimate). "
            "IMPORTANT: This dataset is highly imbalanced (~0.17% fraud). "
            "Use class_weight='balanced' in RandomForestClassifier. "
            "Drop 'id' column if present — it is not a feature."
        ),
    },
    "3": {
        "path": "data/cs-training.csv",
        "desc": (
            "Predict probability of financial distress (Credit Scoring). "
            "Target column: 'SeriousDlqin2yrs' (binary: 1=distress, 0=no distress). "
            "Drop 'Unnamed: 0' — it is a row index, not a feature. "
            "IMPORTANT: Use df.fillna(df.median(numeric_only=True)) instead of dropna() "
            "because MonthlyIncome and NumberOfDependents have ~30K missing values. "
            "Use X = df.drop(columns=['Unnamed: 0', 'SeriousDlqin2yrs']), y = df['SeriousDlqin2yrs']."
        ),
    },
}

# LLM choice names for logging
LLM_NAMES = {
    "1": "groq_llama3.3_70b",
    "2": "gemini-3-flash-preview",
    "3": "ollama",
    "4": "mock",
    "5": "groq_llama4_scout",
    "6": "groq_llama3.1_8b",
    "7": "open router",
}


def select_dataset():
    """Interactive dataset selection."""
    print("\nSelect Dataset:")
    print("1. Credit Card Approval (UCI)")
    print("2. Fraud Detection (Kaggle)")
    print("3. Portfolio Credit Risk (Give Me Some Credit)")
    choice = input("Choice (1-3): ").strip()

    if choice not in DATASETS:
        print("Invalid dataset choice. Exiting.")
        return None, None

    ds = DATASETS[choice]
    return ds["path"], ds["desc"]


def select_mode():
    """Select execution mode."""
    print("\nSelect Execution Mode:")
    print("1. Manual Pipeline (Base Paper - hardcoded agents)")
    print("2. AutoCrew Pipeline (Meta Agent - auto-generated agents)")
    print("3. Comparison Mode (Run BOTH and compare)")
    choice = input("Choice (1-3): ").strip()
    return choice


def main():
    print("=" * 55)
    print("  AutoCrew: Automatic Generation of Agentic AI Crews")
    print("  for Financial Modeling Tasks")
    print("=" * 55)

    # --- Step 1: Select LLM ---
    llm, llm_choice = select_llm_interactive()
    if llm is None:
        sys.exit()

    llm_name = LLM_NAMES.get(llm_choice, "unknown")

    # Show remaining Groq quota at startup (no-op for other providers)
    if llm_choice in ("1", "5", "6"):
        from src.utils.groq_limits import show_groq_limits
        groq_model = getattr(llm, 'model', None)
        show_groq_limits("Start of run", model=groq_model)

    # --- Step 2: Select Dataset ---
    dataset_path, task_desc = select_dataset()
    if dataset_path is None:
        sys.exit()

    # --- Step 3: Select Mode ---
    mode = select_mode()

    # Start capturing terminal output to PDF
    logger = TerminalLogger()
    logger.start()

    try:
        # === MODE 1: Manual (Base Paper) ===
        if mode == "1":
            from src.manual.manual_runner import run_manual_pipeline

            if llm_choice != "4" and not os.path.exists(dataset_path):
                print(f"\nERROR: Dataset not found: {dataset_path}")
                sys.exit()

            results = run_manual_pipeline(llm, dataset_path, task_desc)
            if results:
                print(f"\n[Done] Manual pipeline completed in {results.get('total_time_sec', '?')}s")

        # === MODE 2: AutoCrew (Meta Agent) ===
        elif mode == "2":
            from src.crew_factory.auto_runner import run_auto_pipeline

            if llm_choice != "4" and not os.path.exists(dataset_path):
                print(f"\nERROR: Dataset not found: {dataset_path}")
                sys.exit()

            results = run_auto_pipeline(llm, dataset_path, task_desc, llm_name)
            if results:
                print(f"\n[Done] AutoCrew pipeline completed in {results.get('total_time_sec', '?')}s")

        # === MODE 3: Comparison ===
        elif mode == "3":
            from src.evaluation.comparison import ComparisonRunner

            if llm_choice != "4" and not os.path.exists(dataset_path):
                print(f"\nERROR: Dataset not found: {dataset_path}")
                sys.exit()

            runner = ComparisonRunner()
            runner.run_comparison(llm, dataset_path, task_desc, llm_name)

        else:
            print("Invalid mode. Exiting.")
            sys.exit()

    finally:
        logger.stop_and_save()


if __name__ == "__main__":
    main()
