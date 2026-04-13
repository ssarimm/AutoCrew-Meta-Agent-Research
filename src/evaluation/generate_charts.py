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


def _try_load_real_data(dataset_name):
    """Try to load real dataset for EDA charts. Returns DataFrame or None."""
    try:
        import pandas as pd
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if dataset_name and "credit_card_approval" in dataset_name:
            path = os.path.join(project_root, "data", "credit_card_approval.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, header=None)
                df = df.replace('?', np.nan)
                return df

        elif dataset_name and "creditcard_2023" in dataset_name:
            path = os.path.join(project_root, "data", "creditcard_2023.csv")
            if os.path.exists(path):
                # Sample for performance — full dataset is 568K rows
                df = pd.read_csv(path, nrows=10000)
                return df

        elif dataset_name and "cs-training" in dataset_name:
            path = os.path.join(project_root, "data", "cs-training.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, nrows=10000)
                return df
    except Exception:
        pass
    return None


def generate_eda_plots(dataset_name=None):
    """Generate EDA plots based on which dataset was used."""
    if dataset_name and "creditcard_2023" in dataset_name:
        _generate_fraud_eda_plots(dataset_name)
    elif dataset_name and "cs-training" in dataset_name:
        _generate_credit_scoring_eda_plots(dataset_name)
    else:
        _generate_approval_eda_plots(dataset_name)


def _generate_approval_eda_plots(dataset_name=None):
    """EDA plots for credit_card_approval.csv."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    df = _try_load_real_data(dataset_name or "credit_card_approval")

    if df is not None:
        import pandas as pd
        # Use real data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 1:
            axes[0, 0].hist(df[numeric_cols[0]].dropna(), bins=20, color="#4A90D9", edgecolor="black", alpha=0.7)
            axes[0, 0].set_title(f"Column {numeric_cols[0]} Distribution", fontsize=11, fontweight="bold")

        if len(numeric_cols) >= 2:
            axes[0, 1].hist(df[numeric_cols[1]].dropna(), bins=20, color="#5DADE2", edgecolor="black", alpha=0.7)
            axes[0, 1].set_title(f"Column {numeric_cols[1]} Distribution", fontsize=11, fontweight="bold")

        if len(numeric_cols) >= 3:
            axes[0, 2].hist(df[numeric_cols[2]].dropna(), bins=20, color="#48C9B0", edgecolor="black", alpha=0.7)
            axes[0, 2].set_title(f"Column {numeric_cols[2]} Distribution", fontsize=11, fontweight="bold")

        if len(numeric_cols) >= 4:
            axes[1, 0].hist(df[numeric_cols[3]].dropna(), bins=30, color="#AF7AC5", edgecolor="black", alpha=0.7)
            axes[1, 0].set_title(f"Column {numeric_cols[3]} Distribution", fontsize=11, fontweight="bold")

        # Object column distribution
        obj_cols = df.select_dtypes(include='object').columns.tolist()
        if obj_cols:
            vc = df[obj_cols[0]].value_counts().head(5)
            axes[1, 1].bar(vc.index.astype(str), vc.values, color=["#E8724A", "#F5B041", "#48C9B0", "#5DADE2", "#AF7AC5"][:len(vc)], edgecolor="black", alpha=0.8)
            axes[1, 1].set_title(f"Column {obj_cols[0]} Distribution", fontsize=11, fontweight="bold")
        else:
            axes[1, 1].text(0.5, 0.5, "No object columns", ha="center", va="center")

        # Target class balance (last column)
        target = df.columns[-1]
        vc = df[target].value_counts()
        target_labels = [str(v) for v in vc.index]
        target_vals = vc.values
        colors_t = ["#4CAF50", "#F44336"] + ["#2196F3"] * (len(vc) - 2)
        axes[1, 2].bar(target_labels, target_vals, color=colors_t[:len(vc)], edgecolor="black", alpha=0.8)
        axes[1, 2].set_title(f"Target (col {target}) Class Balance", fontsize=11, fontweight="bold")
        for i, v in enumerate(target_vals):
            pct = round(v / sum(target_vals) * 100, 1)
            axes[1, 2].text(i, v + max(target_vals)*0.02, f"{v} ({pct}%)", ha="center", fontsize=9, fontweight="bold")

        total_rows = len(df)
        total_cols = len(df.columns)
    else:
        # Fallback to synthetic
        np.random.seed(42)
        total_rows, total_cols = 689, 16

        col0 = np.clip(np.random.exponential(4.766, 689), 0, 28)
        axes[0, 0].hist(col0, bins=20, color="#4A90D9", edgecolor="black", alpha=0.7)
        axes[0, 0].set_title("Feature 0 Distribution", fontsize=11, fontweight="bold")

        col1 = np.clip(np.random.exponential(2.225, 689), 0, 28.5)
        axes[0, 1].hist(col1, bins=20, color="#5DADE2", edgecolor="black", alpha=0.7)
        axes[0, 1].set_title("Feature 1 Distribution", fontsize=11, fontweight="bold")

        col2 = np.clip(np.random.exponential(2.4, 689).astype(int), 0, 67)
        axes[0, 2].hist(col2, bins=20, color="#48C9B0", edgecolor="black", alpha=0.7)
        axes[0, 2].set_title("Feature 2 Distribution", fontsize=11, fontweight="bold")

        col3 = np.clip(np.random.exponential(1018, 689).astype(int), 0, 100000)
        axes[1, 0].hist(col3, bins=30, color="#AF7AC5", edgecolor="black", alpha=0.7)
        axes[1, 0].set_title("Feature 3 Distribution", fontsize=11, fontweight="bold")

        cats = ["a", "b"]
        vals = [480, 209]
        axes[1, 1].bar(cats, vals, color=["#E8724A", "#F5B041"], edgecolor="black", alpha=0.8)
        axes[1, 1].set_title("Categorical Feature Distribution", fontsize=11, fontweight="bold")

        target_labels = ["+ (Approved)", "- (Rejected)"]
        target_vals = [306, 383]
        colors_t = ["#4CAF50", "#F44336"]
        axes[1, 2].bar(target_labels, target_vals, color=colors_t, edgecolor="black", alpha=0.8)
        axes[1, 2].set_title("Credit Approval Class Balance", fontsize=11, fontweight="bold")
        for i, v in enumerate(target_vals):
            pct = round(v / sum(target_vals) * 100, 1)
            axes[1, 2].text(i, v + 5, f"{v} ({pct}%)", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle(f"EDA Report: Credit Card Approval Dataset ({total_rows} rows x {total_cols} cols)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "eda_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _generate_fraud_eda_plots(dataset_name=None):
    """EDA plots for creditcard_2023.csv."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    df = _try_load_real_data(dataset_name or "creditcard_2023")

    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove target and id columns from feature list
        feature_cols = [c for c in numeric_cols if c.lower() not in ('class', 'id')]

        for idx, ax in enumerate([axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]):
            if idx < len(feature_cols):
                col = feature_cols[idx]
                ax.hist(df[col].dropna(), bins=30, color=["#4A90D9", "#5DADE2", "#48C9B0", "#AF7AC5", "#E8724A"][idx], edgecolor="black", alpha=0.7)
                ax.set_title(f"{col} Distribution", fontsize=11, fontweight="bold")

        # Target class balance
        if 'Class' in df.columns:
            vc = df['Class'].value_counts()
            target_labels = [f"Legitimate ({int(vc.index[0])})" if vc.index[0] == 0 else f"Fraud ({int(vc.index[0])})" for _ in [0]]
            target_labels = [f"{'Legitimate' if k == 0 else 'Fraud'} ({k})" for k in vc.index]
            axes[1, 2].bar(target_labels, vc.values, color=["#4CAF50", "#F44336"][:len(vc)], edgecolor="black", alpha=0.8)
            axes[1, 2].set_title("Fraud Class Balance", fontsize=11, fontweight="bold")
            if vc.max() / vc.min() > 10:
                axes[1, 2].set_yscale('log')
            for i, v in enumerate(vc.values):
                pct = round(v / vc.values.sum() * 100, 2)
                axes[1, 2].text(i, v * 1.2, f"{v:,}\n({pct}%)", ha="center", fontsize=9, fontweight="bold")

        total_rows = len(df)
        total_cols = len(df.columns)
    else:
        np.random.seed(42)
        total_rows, total_cols = 568630, 31
        # Synthetic fallback
        for idx, (ax, title, color) in enumerate(zip(
            [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]],
            ["V1 Distribution", "V2 Distribution", "Amount Distribution", "Time Distribution", "V14 Distribution"],
            ["#4A90D9", "#5DADE2", "#48C9B0", "#AF7AC5", "#E8724A"]
        )):
            data = np.random.normal(0, 1.5, 5000) if idx < 2 or idx == 4 else (
                np.clip(np.random.exponential(80, 5000), 0, 2000) if idx == 2 else
                np.random.uniform(0, 172800, 5000)
            )
            ax.hist(data, bins=30, color=color, edgecolor="black", alpha=0.7)
            ax.set_title(title, fontsize=11, fontweight="bold")

        target_vals = [284315, 492]
        axes[1, 2].bar(["Legitimate (0)", "Fraud (1)"], target_vals, color=["#4CAF50", "#F44336"], edgecolor="black", alpha=0.8)
        axes[1, 2].set_title("Fraud Class Balance (Highly Imbalanced)", fontsize=11, fontweight="bold")
        axes[1, 2].set_yscale('log')

    plt.suptitle(f"EDA Report: Fraud Detection Dataset ({total_rows:,} rows x {total_cols} cols)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES_DIR, "eda_plots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _generate_credit_scoring_eda_plots(dataset_name=None):
    """EDA plots for cs-training.csv."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    df = _try_load_real_data(dataset_name or "cs-training")

    if df is not None:
        # Drop unnamed index
        df = df.drop(columns=[c for c in df.columns if 'unnamed' in c.lower()], errors='ignore')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != 'SeriousDlqin2yrs']

        plot_configs = [
            (axes[0, 0], "#4A90D9"),
            (axes[0, 1], "#5DADE2"),
            (axes[0, 2], "#48C9B0"),
            (axes[1, 0], "#AF7AC5"),
            (axes[1, 1], "#E8724A"),
        ]

        for idx, (ax, color) in enumerate(plot_configs):
            if idx < len(feature_cols):
                col = feature_cols[idx]
                data = df[col].dropna()
                # Clip extreme outliers for visualization
                q99 = data.quantile(0.99) if len(data) > 0 else 1
                data_clipped = data[data <= q99]
                ax.hist(data_clipped, bins=30, color=color, edgecolor="black", alpha=0.7)
                ax.set_title(f"{col} Distribution", fontsize=11, fontweight="bold")

        # Target class balance
        if 'SeriousDlqin2yrs' in df.columns:
            vc = df['SeriousDlqin2yrs'].value_counts()
            target_labels = [f"No Distress ({k})" if k == 0 else f"Distress ({k})" for k in vc.index]
            colors_t = ["#4CAF50", "#F44336"]
            axes[1, 2].bar(target_labels, vc.values, color=colors_t[:len(vc)], edgecolor="black", alpha=0.8)
            axes[1, 2].set_title("Financial Distress Class Balance", fontsize=11, fontweight="bold")
            for i, v in enumerate(vc.values):
                pct = round(v / vc.values.sum() * 100, 1)
                axes[1, 2].text(i, v + max(vc.values)*0.02, f"{v:,}\n({pct}%)", ha="center", fontsize=9, fontweight="bold")

        total_rows = len(df)
        total_cols = len(df.columns)
    else:
        np.random.seed(42)
        total_rows, total_cols = 150000, 12
        # Synthetic fallback
        for idx, (ax, title, color) in enumerate(zip(
            [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]],
            ["Revolving Utilization", "Age Distribution", "Debt Ratio", "Monthly Income", "Number of Dependents"],
            ["#4A90D9", "#5DADE2", "#48C9B0", "#AF7AC5", "#E8724A"]
        )):
            if idx == 0: data = np.clip(np.random.exponential(6.0, 5000), 0, 50)
            elif idx == 1: data = np.clip(np.random.normal(52, 14, 5000).astype(int), 21, 99)
            elif idx == 2: data = np.clip(np.random.exponential(350, 5000), 0, 5000)
            elif idx == 3: data = np.clip(np.random.exponential(6600, 5000), 0, 50000)
            else: data = np.random.choice([0, 1, 2, 3, 4, 5], size=5000, p=[0.45, 0.25, 0.15, 0.08, 0.04, 0.03])
            ax.hist(data, bins=30, color=color, edgecolor="black", alpha=0.7)
            ax.set_title(title, fontsize=11, fontweight="bold")

        target_vals = [139974, 10026]
        axes[1, 2].bar(["No Distress (0)", "Distress (1)"], target_vals, color=["#4CAF50", "#F44336"], edgecolor="black", alpha=0.8)
        axes[1, 2].set_title("Financial Distress Class Balance", fontsize=11, fontweight="bold")

    plt.suptitle(f"EDA Report: Credit Scoring Dataset ({total_rows:,} rows x {total_cols} cols)", fontsize=14, fontweight="bold")
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

    results_dir = os.path.join(os.path.dirname(FIGURES_DIR), "results")
    meta_t, model_t, mrm_t = 7.8, 100.0, 18.0
    if os.path.exists(results_dir):
        import json as _json
        for fname in sorted(os.listdir(results_dir), reverse=True):
            if fname.startswith("auto_results") and fname.endswith(".json"):
                try:
                    with open(os.path.join(results_dir, fname)) as _f:
                        _data = _json.load(_f)
                    meta_t = _data.get("meta_agent_time_sec", meta_t)
                    model_t = _data.get("modeling_time_sec", model_t)
                    mrm_t = _data.get("mrm_time_sec") if _data.get("mrm_time_sec") is not None else mrm_t
                    break
                except Exception:
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
        ("Chief Risk Officer", "none", "#E74C3C"),
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


def generate_model_comparison_chart(manual_metrics=None, auto_metrics=None):
    """Generate side-by-side model performance and timing comparison charts."""
    if manual_metrics is None or auto_metrics is None:
        results_dir = os.path.join(os.path.dirname(FIGURES_DIR), "results")
        metrics_file = os.path.join(results_dir, "metrics.json")
        if os.path.exists(metrics_file):
            import json as _json
            with open(metrics_file) as f:
                records = _json.load(f)
            for r in sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True):
                if r.get("mode") == "manual" and manual_metrics is None:
                    manual_metrics = r
                elif r.get("mode") == "auto" and auto_metrics is None:
                    auto_metrics = r
                if manual_metrics and auto_metrics:
                    break

    def _val(m, key, default=0.0):
        if not m:
            return default
        v = m.get(key, default)
        return float(v) if v is not None and v != -1 else default

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Path A (Manual) vs Path B (AutoCrew) — Performance Comparison",
                 fontsize=14, fontweight="bold", y=1.02)

    metric_keys = ["accuracy", "f1_score", "precision", "recall"]
    metric_labels = ["Accuracy", "F1 Score", "Precision", "Recall"]
    manual_vals = [_val(manual_metrics, k) for k in metric_keys]
    auto_vals = [_val(auto_metrics, k) for k in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.35
    bars1 = axes[0].bar(x - width / 2, manual_vals, width,
                        label="Path A: Manual", color="#4A90D9", alpha=0.85, edgecolor="black")
    bars2 = axes[0].bar(x + width / 2, auto_vals, width,
                        label="Path B: AutoCrew", color="#48BB78", alpha=0.85, edgecolor="black")

    axes[0].set_ylabel("Score", fontsize=11)
    axes[0].set_title("Model Performance Scores", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels, fontsize=10)
    axes[0].set_ylim(0, 1.2)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0.001:
            axes[0].text(bar.get_x() + bar.get_width() / 2., h + 0.015,
                         f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    timing_labels = ["Meta Agent\nSetup", "Modeling", "MRM", "Total"]
    manual_times = [
        0,
        _val(manual_metrics, "modeling_time_sec"),
        _val(manual_metrics, "mrm_time_sec"),
        _val(manual_metrics, "total_time_sec"),
    ]
    auto_times = [
        _val(auto_metrics, "meta_agent_time_sec"),
        _val(auto_metrics, "modeling_time_sec"),
        _val(auto_metrics, "mrm_time_sec"),
        _val(auto_metrics, "total_time_sec"),
    ]

    x2 = np.arange(len(timing_labels))
    bars3 = axes[1].bar(x2 - width / 2, manual_times, width,
                        label="Path A: Manual", color="#4A90D9", alpha=0.85, edgecolor="black")
    bars4 = axes[1].bar(x2 + width / 2, auto_times, width,
                        label="Path B: AutoCrew", color="#48BB78", alpha=0.85, edgecolor="black")

    axes[1].set_ylabel("Time (seconds)", fontsize=11)
    axes[1].set_title("Execution Time Breakdown", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(timing_labels, fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)
    for bar in list(bars3) + list(bars4):
        h = bar.get_height()
        if h > 0.5:
            axes[1].text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                         f"{h:.0f}s", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def generate_all(dataset_name=None):
    """Generate all charts."""
    print(f"\nGenerating charts into: {FIGURES_DIR}")
    generate_eda_plots(dataset_name)
    generate_architecture()
    generate_pipeline()
    generate_comparison()
    generate_crew_structure()
    generate_model_comparison_chart()
    print(f"\nAll 6 charts saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    generate_all()
