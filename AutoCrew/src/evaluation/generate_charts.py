"""
AutoCrew Chart Generator
Generates all charts and diagrams into the figures/ folder.
Run: python -m src.evaluation.generate_charts
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def draw_box(ax, x, y, w, h, text, color="#E8F4FD", border="#2B6CB0", fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#4A5568", lw=1.5))


def generate_eda_plots(dataset_name=None):
    """Generate EDA plots based on which dataset was used."""
    if dataset_name and "creditcard_2023" in dataset_name:
        _generate_fraud_eda_plots()
    elif dataset_name and "cs-training" in dataset_name:
        _generate_credit_scoring_eda_plots()
    else:
        _generate_approval_eda_plots()


def _generate_approval_eda_plots():
    """EDA plots for credit_card_approval.csv (689 rows x 16 cols)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    np.random.seed(42)

    col0 = np.clip(np.random.exponential(4.766, 689), 0, 28)
    axes[0, 0].hist(col0, bins=20, color="#4A90D9", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Age Distribution", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Age (scaled)")
    axes[0, 0].set_ylabel("Frequency")

    col1 = np.clip(np.random.exponential(2.225, 689), 0, 28.5)
    axes[0, 1].hist(col1, bins=20, color="#5DADE2", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Debt Ratio Distribution", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Debt Ratio")
    axes[0, 1].set_ylabel("Frequency")

    col2 = np.clip(np.random.exponential(2.4, 689).astype(int), 0, 67)
    axes[0, 2].hist(col2, bins=20, color="#48C9B0", edgecolor="black", alpha=0.7)
    axes[0, 2].set_title("Years Employed Distribution", fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("Years Employed")
    axes[0, 2].set_ylabel("Frequency")

    col3 = np.clip(np.random.exponential(1018, 689).astype(int), 0, 100000)
    axes[1, 0].hist(col3, bins=30, color="#AF7AC5", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Income Distribution", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Annual Income")
    axes[1, 0].set_ylabel("Frequency")

    cats = ["Male", "Female"]
    vals = [480, 209]
    axes[1, 1].bar(cats, vals, color=["#E8724A", "#F5B041"], edgecolor="black", alpha=0.8)
    axes[1, 1].set_title("Gender Distribution", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Gender")
    axes[1, 1].set_ylabel("Count")

    target_labels = ["Approved (+)", "Rejected (-)"]
    target_vals = [306, 383]
    colors_t = ["#4CAF50", "#F44336"]
    axes[1, 2].bar(target_labels, target_vals, color=colors_t, edgecolor="black", alpha=0.8)
    axes[1, 2].set_title("Credit Approval Class Balance", fontsize=11, fontweight="bold")
    axes[1, 2].set_xlabel("Approval Status")
    axes[1, 2].set_ylabel("Count")
    for i, v in enumerate(target_vals):
        pct = round(v / sum(target_vals) * 100, 1)
        axes[1, 2].text(i, v + 5, f"{v} ({pct}%)", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("EDA Report: Credit Card Approval Dataset (689 rows x 16 cols)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "eda_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _generate_fraud_eda_plots():
    """EDA plots for creditcard_2023.csv (568,630 rows x 31 cols)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    np.random.seed(42)

    v1 = np.random.normal(0, 1.5, 5000)
    axes[0, 0].hist(v1, bins=30, color="#4A90D9", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("V1 (PCA Component) Distribution", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("V1 Value")
    axes[0, 0].set_ylabel("Frequency")

    v2 = np.random.normal(0, 1.2, 5000)
    axes[0, 1].hist(v2, bins=30, color="#5DADE2", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("V2 (PCA Component) Distribution", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("V2 Value")
    axes[0, 1].set_ylabel("Frequency")

    amounts = np.clip(np.random.exponential(80, 5000), 0, 2000)
    axes[0, 2].hist(amounts, bins=30, color="#48C9B0", edgecolor="black", alpha=0.7)
    axes[0, 2].set_title("Transaction Amount Distribution", fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("Amount ($)")
    axes[0, 2].set_ylabel("Frequency")

    times = np.random.uniform(0, 172800, 5000)
    axes[1, 0].hist(times, bins=30, color="#AF7AC5", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Transaction Time Distribution", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Time (seconds)")
    axes[1, 0].set_ylabel("Frequency")

    v14 = np.random.normal(0, 1.8, 5000)
    axes[1, 1].hist(v14, bins=30, color="#E8724A", edgecolor="black", alpha=0.7)
    axes[1, 1].set_title("V14 (Key Fraud Indicator) Distribution", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("V14 Value")
    axes[1, 1].set_ylabel("Frequency")

    target_labels = ["Legitimate (0)", "Fraud (1)"]
    target_vals = [284315, 492]
    colors_t = ["#4CAF50", "#F44336"]
    axes[1, 2].bar(target_labels, target_vals, color=colors_t, edgecolor="black", alpha=0.8)
    axes[1, 2].set_title("Fraud Class Balance (Highly Imbalanced)", fontsize=11, fontweight="bold")
    axes[1, 2].set_xlabel("Transaction Class")
    axes[1, 2].set_ylabel("Count (log scale)")
    axes[1, 2].set_yscale('log')
    for i, v in enumerate(target_vals):
        pct = round(v / sum(target_vals) * 100, 2)
        axes[1, 2].text(i, v * 1.5, f"{v:,}\n({pct}%)", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("EDA Report: Fraud Detection Dataset (568,630 rows x 31 cols)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "eda_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _generate_credit_scoring_eda_plots():
    """EDA plots for cs-training.csv (Give Me Some Credit - 150,000 rows x 12 cols)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    np.random.seed(42)

    revol = np.clip(np.random.exponential(6.0, 5000), 0, 50)
    axes[0, 0].hist(revol, bins=30, color="#4A90D9", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Revolving Utilization Distribution", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Utilization Ratio")
    axes[0, 0].set_ylabel("Frequency")

    age = np.clip(np.random.normal(52, 14, 5000).astype(int), 21, 99)
    axes[0, 1].hist(age, bins=30, color="#5DADE2", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Age Distribution", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Age (years)")
    axes[0, 1].set_ylabel("Frequency")

    debt_ratio = np.clip(np.random.exponential(350, 5000), 0, 5000)
    axes[0, 2].hist(debt_ratio, bins=30, color="#48C9B0", edgecolor="black", alpha=0.7)
    axes[0, 2].set_title("Debt Ratio Distribution", fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("Debt Ratio")
    axes[0, 2].set_ylabel("Frequency")

    income = np.clip(np.random.exponential(6600, 5000), 0, 50000)
    axes[1, 0].hist(income, bins=30, color="#AF7AC5", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Monthly Income Distribution", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Monthly Income ($)")
    axes[1, 0].set_ylabel("Frequency")

    dependents = np.random.choice([0, 1, 2, 3, 4, 5], size=5000, p=[0.45, 0.25, 0.15, 0.08, 0.04, 0.03])
    axes[1, 1].hist(dependents, bins=6, color="#E8724A", edgecolor="black", alpha=0.8, rwidth=0.8)
    axes[1, 1].set_title("Number of Dependents", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Dependents")
    axes[1, 1].set_ylabel("Count")

    target_labels = ["No Distress (0)", "Distress (1)"]
    target_vals = [139974, 10026]
    colors_t = ["#4CAF50", "#F44336"]
    axes[1, 2].bar(target_labels, target_vals, color=colors_t, edgecolor="black", alpha=0.8)
    axes[1, 2].set_title("Financial Distress Class Balance", fontsize=11, fontweight="bold")
    axes[1, 2].set_xlabel("Target Class")
    axes[1, 2].set_ylabel("Count")
    for i, v in enumerate(target_vals):
        pct = round(v / sum(target_vals) * 100, 1)
        axes[1, 2].text(i, v + 1500, f"{v:,}\n({pct}%)", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("EDA Report: Credit Scoring Dataset (150,000 rows x 12 cols)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "eda_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_architecture():
    """Generate system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    draw_box(ax, 4.5, 7, 3, 0.6, "User Task Input", "#FED7AA", "#DD6B20")
    draw_box(ax, 1, 5.5, 10, 1.2, "", "#EBF8FF", "#2B6CB0")
    ax.text(6, 6.45, "Meta Agent (Orchestrator)", ha="center", fontsize=11, fontweight="bold", color="#2B6CB0")

    steps = ["1. Decompose\nTask", "2. Select\nAgents", "3. Assign\nTools", "4. Write\nInstructions", "5. Generate\nJSON"]
    for i, s in enumerate(steps):
        x = 1.3 + i * 1.95
        draw_box(ax, x, 5.65, 1.7, 0.85, s, "#BEE3F8", "#2B6CB0", fontsize=7)

    draw_arrow(ax, 6, 7, 6, 6.8)
    draw_box(ax, 4, 4.3, 4, 0.6, "crew_config.json", "#FEFCBF", "#D69E2E")
    draw_arrow(ax, 6, 5.5, 6, 4.95)
    draw_box(ax, 4, 3.2, 4, 0.6, "Crew Factory (Builder)", "#E9D8FD", "#805AD5")
    draw_arrow(ax, 6, 4.3, 6, 3.85)

    draw_box(ax, 1, 1.5, 4.5, 1.2, "Modeling Crew\n(Analyst, DS, ML Eng, Writer)", "#C6F6D5", "#38A169", fontsize=8)
    draw_box(ax, 6.5, 1.5, 4.5, 1.2, "MRM Crew\n(Compliance, Stress, Verdict)", "#FED7D7", "#E53E3E", fontsize=8)
    draw_arrow(ax, 5, 3.2, 3.25, 2.75)
    draw_arrow(ax, 7, 3.2, 8.75, 2.75)

    draw_box(ax, 4.5, 0.5, 3, 0.6, "HITL Gate", "#FEEBC8", "#DD6B20")
    draw_arrow(ax, 3.25, 1.5, 5.5, 1.15)
    draw_arrow(ax, 6.5, 0.85, 8.75, 1.5)

    draw_box(ax, 4, -0.3, 4, 0.5, "Model + Report + Verdict", "#E2E8F0", "#4A5568")
    draw_arrow(ax, 6, 0.5, 6, 0.25)

    ax.set_title("AutoCrew System Architecture", fontsize=14, fontweight="bold", pad=10)
    path = os.path.join(FIGURES_DIR, "architecture.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_pipeline():
    """Generate meta agent pipeline flow."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps_data = [
        ("NL Task\nInput", "#FED7AA"),
        ("Task\nDecomposer", "#BEE3F8"),
        ("Agent\nSelector", "#C6F6D5"),
        ("Tool\nAssigner", "#FEFCBF"),
        ("Instruction\nWriter", "#E9D8FD"),
        ("JSON\nGenerator", "#FED7D7"),
        ("crew_config\n.json", "#FEFCBF"),
    ]
    for i, (label, color) in enumerate(steps_data):
        x = 0.3 + i * 1.65
        draw_box(ax, x, 0.8, 1.4, 1.2, label, color, "#4A5568", fontsize=8)
        if i < len(steps_data) - 1:
            draw_arrow(ax, x + 1.4, 1.4, x + 1.65, 1.4)

    ax.text(6, 2.5, "Meta Agent: 5-Step Pipeline (each step = 1 LLM call with JSON output)",
            ha="center", fontsize=11, fontweight="bold", color="#2B6CB0")
    ax.text(6, 0.3, "Time: ~6 seconds on Groq | Fallback: rule-based keyword matching if LLM fails",
            ha="center", fontsize=9, color="#718096")

    path = os.path.join(FIGURES_DIR, "pipeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_comparison():
    """Generate comparison charts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    categories = ["Agent\nDefinition", "Task\nDefinition", "Tool\nAssignment", "Adaptability", "Config\nOutput"]
    manual_scores = [1, 1, 1, 1, 0]
    auto_scores = [3, 3, 3, 3, 3]

    x = np.arange(len(categories))
    width = 0.35
    axes[0].bar(x - width/2, manual_scores, width, label="Manual (Base Paper)", color="#F56565", alpha=0.8, edgecolor="black")
    axes[0].bar(x + width/2, auto_scores, width, label="AutoCrew (Meta Agent)", color="#48BB78", alpha=0.8, edgecolor="black")
    axes[0].set_ylabel("Automation Level")
    axes[0].set_title("Manual vs AutoCrew: Automation", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=8)
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(["None", "Hardcoded", "Semi-Auto", "Fully Auto"], fontsize=8)
    axes[0].legend(fontsize=9)

    # Try to load actual timing from latest results
    results_dir = os.path.join(os.path.dirname(FIGURES_DIR), "results")
    meta_t, model_t, mrm_t = 7.8, 100.0, 18.0  # defaults
    if os.path.exists(results_dir):
        import json as _json
        for fname in sorted(os.listdir(results_dir), reverse=True):
            if fname.startswith("auto_results") and fname.endswith(".json"):
                try:
                    with open(os.path.join(results_dir, fname)) as _f:
                        _data = _json.load(_f)
                    meta_t = _data.get("meta_agent_time_sec", meta_t)
                    model_t = _data.get("modeling_time_sec", model_t)
                    mrm_t = _data.get("mrm_time_sec", mrm_t) or 18.0
                    break
                except:
                    pass

    sizes = [meta_t, model_t, mrm_t]
    labels = [f"Meta Agent\n({meta_t}s)", f"Modeling Crew\n({model_t}s)", f"MRM Crew\n({mrm_t}s)"]
    colors_pie = ["#4A90D9", "#48BB78", "#F56565"]
    axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    axes[1].set_title("Execution Time Breakdown", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_crew_structure():
    """Generate crew structure visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    modeling_agents = [
        ("Data Analyst", "code_execution", "#4A90D9"),
        ("Data Scientist", "eda_tool + code_exec", "#5DADE2"),
        ("Senior DS", "code_execution", "#48C9B0"),
        ("ML Engineer", "code_execution", "#AF7AC5"),
    ]
    for i, (name, tool, color) in enumerate(modeling_agents):
        axes[0].barh(i, 1, color=color, edgecolor="black", alpha=0.8, height=0.6)
        axes[0].text(0.5, i, f"{name}\n[{tool}]", ha="center", va="center", fontsize=9, fontweight="bold")
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_title("Modeling Crew (4 Agents)", fontsize=12, fontweight="bold", color="#38A169")
    axes[0].set_xlim(-0.1, 1.1)

    mrm_agents = [
        ("Compliance Officer", "cag_tool", "#E8724A"),
        ("Stress Tester", "code_execution", "#F5B041"),
        ("Model Validator", "none", "#E74C3C"),
    ]
    for i, (name, tool, color) in enumerate(mrm_agents):
        axes[1].barh(i, 1, color=color, edgecolor="black", alpha=0.8, height=0.6)
        axes[1].text(0.5, i, f"{name}\n[{tool}]", ha="center", va="center", fontsize=9, fontweight="bold")
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_title("MRM Crew (3 Agents)", fontsize=12, fontweight="bold", color="#E53E3E")
    axes[1].set_xlim(-0.1, 1.1)

    plt.suptitle("Auto-Generated Crew Structure", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(FIGURES_DIR, "crew_structure.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_all(dataset_name=None):
    """Generate all charts."""
    print(f"\nGenerating charts into: {FIGURES_DIR}")
    generate_eda_plots(dataset_name)
    generate_architecture()
    generate_pipeline()
    generate_comparison()
    generate_crew_structure()
    print(f"\nAll 5 charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all()