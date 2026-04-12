"""
AutoCrew PDF Report Generator
Focused on Path A (Manual) vs Path B (AutoCrew) comparison.
"""
import os
import re
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)
from reportlab.lib.enums import TA_CENTER

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# ── Human setup time estimate ──────────────────────────────────────────────────
# For Path A (Manual), a developer must hand-code every agent definition and task
# instruction in Python before a single run can happen.  This one-time effort is
# invisible in wall-clock run timings but is a real cost that AutoCrew eliminates.
#
# Estimation breakdown (experienced CrewAI developer):
#   - 6 agent definitions (role, goal, backstory, tool selection) @ 15 min each → 90 min
#   - 6 task descriptions (detailed LLM-quality instructions)     @ 20 min each → 120 min
#   - Tool registry wiring + pipeline integration                              →  30 min
#   - Testing, debugging, and prompt iteration                                 →  60 min
#   ─────────────────────────────────────────────────────────────────────────────────────
#   Total estimate                                                             → 300 min
#
HUMAN_SETUP_TIME_SEC = 300 * 60  # 300 minutes = 18 000 s
HUMAN_SETUP_LABEL    = "~300 min (est.)"  # shown in tables


def _get_dataset_display_name(path):
    if "creditcard_2023" in path:
        return "Fraud Detection — creditcard_2023.csv"
    elif "cs-training" in path:
        return "Credit Scoring — cs-training.csv (Give Me Some Credit)"
    elif "credit_card_approval" in path:
        return "Credit Card Approval — credit_card_approval.csv"
    return path or "Unknown dataset"


def _extract_metric(text: str, metric_name: str) -> float:
    """Extract a numeric metric from output text. Returns -1 if not found."""
    patterns = [
        rf"\*\*{re.escape(metric_name)}\*\*\s*[:=]?\s*([\d.]+)",
        rf"{re.escape(metric_name)}\s*[:=]\s*([\d.]+)",
        rf"{re.escape(metric_name)}.*?(0\.\d{{2,4}})",
    ]
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                val = float(match.group(1))
                if val > 1:
                    val /= 100.0
                if 0 < val <= 1:
                    return round(val, 4)
            except ValueError:
                continue
    return -1.0


def _extract_all_metrics(results: dict) -> dict:
    """Extract ML metrics and timing from a pipeline results dict."""
    text = results.get("modeling_output", "") or ""
    return {
        "accuracy":  _extract_metric(text, "accuracy"),
        "f1_score":  _extract_metric(text, "f1"),
        "precision": _extract_metric(text, "precision"),
        "recall":    _extract_metric(text, "recall"),
        "meta_agent_time_sec": results.get("meta_agent_time_sec", 0) or 0,
        "modeling_time_sec":   results.get("modeling_time_sec", 0) or 0,
        "mrm_time_sec":        results.get("mrm_time_sec", 0) or 0,
        "total_time_sec":      results.get("total_time_sec", 0) or 0,
        "mrm_verdict": results.get("mrm_verdict", "N/A"),
        "mode": results.get("mode", "unknown"),
    }


def _load_latest_metrics_from_file():
    """Load manual and auto metrics from the most recent results JSON files."""
    manual, auto = {}, {}
    results_dir = os.path.join(PROJECT_ROOT, "results")
    if not os.path.exists(results_dir):
        return manual, auto

    # Check metrics.json first
    mfile = os.path.join(results_dir, "metrics.json")
    if os.path.exists(mfile):
        with open(mfile) as f:
            records = json.load(f)
        for r in sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True):
            if r.get("mode") == "manual" and not manual:
                manual = r
            elif r.get("mode") == "auto" and not auto:
                auto = r
            if manual and auto:
                break

    # Fall back to individual result files
    if not manual:
        for fname in sorted(os.listdir(results_dir), reverse=True):
            if fname.startswith("manual_results") and fname.endswith(".json"):
                with open(os.path.join(results_dir, fname)) as f:
                    raw = json.load(f)
                manual = _extract_all_metrics(raw)
                manual["mode"] = "manual"
                break
    if not auto:
        for fname in sorted(os.listdir(results_dir), reverse=True):
            if fname.startswith("auto_results") and fname.endswith(".json"):
                with open(os.path.join(results_dir, fname)) as f:
                    raw = json.load(f)
                auto = _extract_all_metrics(raw)
                auto["mode"] = "auto"
                break

    return manual, auto


def generate_report(
    mode="auto",
    results=None,
    output_path="results/AutoCrew_Report.pdf",
    **_,   # absorb legacy config_path= calls without error
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build metrics for the current run
    current_metrics = _extract_all_metrics(results) if results else {}

    # Load previous run metrics from file for the other path
    saved_manual, saved_auto = _load_latest_metrics_from_file()

    if mode == "manual":
        manual_metrics = current_metrics
        auto_metrics = saved_auto
    else:
        manual_metrics = saved_manual
        auto_metrics = current_metrics

    dataset_name = (results.get("dataset", "") if results else "") or \
                   manual_metrics.get("dataset", "") or auto_metrics.get("dataset", "")

    # Generate / update all charts for this run
    print("[Report] Generating charts...")
    chart_path = None
    eda_path = None
    arch_path = os.path.join(FIGURES_DIR, "architecture.png")
    pipeline_path = os.path.join(FIGURES_DIR, "pipeline.png")
    comparison_path = os.path.join(FIGURES_DIR, "comparison.png")
    crew_path = os.path.join(FIGURES_DIR, "crew_structure.png")

    try:
        from src.evaluation.generate_charts import (
            generate_model_comparison_chart,
            generate_eda_plots,
            generate_architecture,
            generate_pipeline,
            generate_comparison,
            generate_crew_structure,
        )
        chart_path = generate_model_comparison_chart(
            manual_metrics if manual_metrics else None,
            auto_metrics if auto_metrics else None,
        )
        generate_eda_plots(dataset_name)
        eda_path = os.path.join(FIGURES_DIR, "eda_plots.png")
        generate_architecture()
        generate_pipeline()
        generate_comparison()
        generate_crew_structure()
    except Exception as e:
        print(f"[Report] Chart generation failed: {e}")
        # Fall back to whatever already exists on disk
        eda_path = os.path.join(FIGURES_DIR, "eda_plots.png") if os.path.exists(os.path.join(FIGURES_DIR, "eda_plots.png")) else None

    # ── PDF setup ──────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=1.5 * cm, leftMargin=1.5 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle("MainTitle", parent=styles["Title"],
        fontSize=20, spaceAfter=6, textColor=colors.HexColor("#1a1a2e")))
    styles.add(ParagraphStyle("SubTitle", parent=styles["Normal"],
        fontSize=11, alignment=TA_CENTER,
        textColor=colors.HexColor("#666666"), spaceAfter=16))
    styles.add(ParagraphStyle("SectionHead", parent=styles["Heading1"],
        fontSize=14, textColor=colors.HexColor("#16213e"),
        spaceBefore=18, spaceAfter=8,
        borderWidth=1, borderColor=colors.HexColor("#0f3460"), borderPadding=4))
    styles.add(ParagraphStyle("BodyText2", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle("Caption", parent=styles["Normal"],
        fontSize=9, alignment=TA_CENTER,
        textColor=colors.HexColor("#888888"), spaceAfter=10))
    styles.add(ParagraphStyle("VerdictApproved", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#276749"), spaceAfter=4))
    styles.add(ParagraphStyle("VerdictRejected", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#c53030"), spaceAfter=4))

    story = []

    def make_table(data, widths, header_color="#16213e"):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        return t

    def fmt_metric(v):
        """Format a metric value or return N/A."""
        try:
            f = float(v)
            return f"{f:.4f}" if f >= 0 else "N/A"
        except (TypeError, ValueError):
            return "N/A"

    def fmt_time(v):
        try:
            f = float(v)
            return f"{f:.1f}s" if f > 0 else "—"
        except (TypeError, ValueError):
            return "—"

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1: COVER
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("AutoCrew", styles["MainTitle"]))
    story.append(Paragraph(
        "Path A (Manual) vs Path B (AutoCrew) — Comparison Report",
        styles["SubTitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}",
        styles["SubTitle"]))
    story.append(Paragraph(
        f"Dataset: <b>{_get_dataset_display_name(dataset_name)}</b>",
        styles["SubTitle"]))
    story.append(Spacer(1, 0.4 * inch))

    cover_data = [
        ["Path", "Method", "Status"],
        ["Path A — Manual", "Hardcoded agents/tasks (base paper)",
         "Metrics loaded" if any(v > 0 for v in [manual_metrics.get("accuracy", -1)]) else "No data"],
        ["Path B — AutoCrew", "Meta Agent auto-generates crew from NL",
         "Metrics loaded" if any(v > 0 for v in [auto_metrics.get("accuracy", -1)]) else "No data"],
    ]
    story.append(make_table(cover_data, [3.5 * cm, 9 * cm, 4 * cm], "#1a1a2e"))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2: MODEL PERFORMANCE COMPARISON  (KEY PAGE)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. Model Performance Comparison", styles["SectionHead"]))
    story.append(Paragraph(
        "Both paths train a <b>Random Forest Classifier</b> on the same dataset. "
        "The table below shows the test-set metrics for each path.",
        styles["BodyText2"]))

    perf_data = [
        ["Metric", "Path A: Manual", "Path B: AutoCrew", "Difference"],
    ]
    for label, key in [("Accuracy", "accuracy"), ("F1 Score", "f1_score"),
                       ("Precision", "precision"), ("Recall", "recall")]:
        mv = manual_metrics.get(key, -1)
        av = auto_metrics.get(key, -1)
        if mv is not None and mv < 0:
            mv = -1
        if av is not None and av < 0:
            av = -1
        mv_f = float(mv) if mv not in (None, -1) else None
        av_f = float(av) if av not in (None, -1) else None
        if mv_f is not None and av_f is not None:
            diff = f"{av_f - mv_f:+.4f}"
        else:
            diff = "N/A"
        perf_data.append([label, fmt_metric(mv), fmt_metric(av), diff])

    t = make_table(perf_data, [3.5 * cm, 4.5 * cm, 4.5 * cm, 4 * cm], "#0f3460")
    # Highlight difference column
    for row_idx in range(1, len(perf_data)):
        diff_val = perf_data[row_idx][3]
        if diff_val.startswith("+") and diff_val != "N/A":
            t.setStyle(TableStyle([
                ("TEXTCOLOR", (3, row_idx), (3, row_idx), colors.HexColor("#276749")),
                ("FONTNAME", (3, row_idx), (3, row_idx), "Helvetica-Bold"),
            ]))
        elif diff_val.startswith("-"):
            t.setStyle(TableStyle([
                ("TEXTCOLOR", (3, row_idx), (3, row_idx), colors.HexColor("#c53030")),
                ("FONTNAME", (3, row_idx), (3, row_idx), "Helvetica-Bold"),
            ]))
    story.append(t)
    story.append(Spacer(1, 0.2 * inch))

    # Embed the comparison bar chart
    if chart_path and os.path.exists(chart_path):
        story.append(Image(chart_path, width=16 * cm, height=7 * cm))
        story.append(Paragraph(
            "Figure 1: Model performance scores and execution time — Manual vs AutoCrew",
            styles["Caption"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: DATASET ANALYSIS (EDA)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Dataset Analysis", styles["SectionHead"]))
    story.append(Paragraph(
        f"Exploratory data analysis of <b>{_get_dataset_display_name(dataset_name)}</b>. "
        "Charts show feature distributions, class balance, and key statistics. "
        "Both paths train on the same unmodified dataset.",
        styles["BodyText2"]))
    if eda_path and os.path.exists(eda_path):
        story.append(Spacer(1, 0.1 * inch))
        story.append(Image(eda_path, width=16 * cm, height=9.5 * cm))
        story.append(Paragraph(
            "Figure 2: EDA — feature distributions and class balance",
            styles["Caption"]))
    else:
        story.append(Paragraph("<i>EDA chart not available.</i>", styles["BodyText2"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # AGENT-GENERATED PLOTS (inserted if any exist)
    # ══════════════════════════════════════════════════════════════════════════
    agent_plots_dir = os.path.join(PROJECT_ROOT, "figures", "agent_plots")
    agent_plot_files = []
    if os.path.exists(agent_plots_dir):
        agent_plot_files = sorted([
            os.path.join(agent_plots_dir, f)
            for f in os.listdir(agent_plots_dir)
            if f.endswith(".png")
        ])

    if agent_plot_files:
        story.append(Paragraph("3. Agent-Generated Plots", styles["SectionHead"]))
        story.append(Paragraph(
            "The following charts were automatically generated by agents during code execution.",
            styles["BodyText2"]))

        for i, plot_path in enumerate(agent_plot_files):
            # Derive a human-readable label from the filename:
            # format is  <timestamp>_<sanitised_title>.png
            fname = os.path.basename(plot_path).replace(".png", "")
            parts = fname.split("_", 1)          # split off timestamp
            raw_label = parts[1] if len(parts) > 1 else fname
            label = raw_label.replace("_", " ").strip().title() or f"Plot {i + 1}"

            try:
                # Fit each plot to the page width; preserve aspect ratio
                from PIL import Image as PILImage
                with PILImage.open(plot_path) as img:
                    w_px, h_px = img.size
                max_w = 15 * cm
                aspect = h_px / w_px
                img_w = max_w
                img_h = min(max_w * aspect, 10 * cm)
            except Exception:
                img_w, img_h = 15 * cm, 9 * cm

            story.append(Spacer(1, 0.15 * inch))
            story.append(Image(plot_path, width=img_w, height=img_h))
            story.append(Paragraph(
                f"Figure {3 + i}: {label}",
                styles["Caption"]))

        story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3: EXECUTION TIMING
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. Execution Time Comparison", styles["SectionHead"]))
    story.append(Paragraph(
        "Wall-clock run time is measured automatically. "
        "Path A also incurs a one-time <b>human setup cost</b> — the engineering effort "
        "required to hand-code every agent definition, task instruction, and tool "
        "assignment before the pipeline can execute. "
        "Path B eliminates this cost entirely: the Meta Agent generates the full crew "
        "configuration from a single natural-language sentence.",
        styles["BodyText2"]))
    story.append(Spacer(1, 0.1 * inch))

    manual_total = manual_metrics.get("total_time_sec", 0) or 0
    auto_total   = auto_metrics.get("total_time_sec", 0) or 0

    def fmt_total_with_setup(run_sec):
        """Show run time + human setup as a combined total."""
        if run_sec <= 0:
            return f"— + {HUMAN_SETUP_LABEL}"
        combined_min = (run_sec + HUMAN_SETUP_TIME_SEC) / 60
        return f"{fmt_time(run_sec)} + {HUMAN_SETUP_LABEL}\n= ~{combined_min:.0f} min total"

    timing_data = [
        ["Phase", "Path A: Manual", "Path B: AutoCrew", "Notes"],
        ["Human Setup\n(one-time, estimated)",
         HUMAN_SETUP_LABEL,
         "~6 s (automated)",
         "Write agent defs, task instructions,\ntool assignments, debug & iterate"],
        ["Meta Agent Config", "—",
         fmt_time(auto_metrics.get("meta_agent_time_sec")),
         "LLM crew generation (AutoCrew only)"],
        ["Modeling Crew",
         fmt_time(manual_metrics.get("modeling_time_sec")),
         fmt_time(auto_metrics.get("modeling_time_sec")),
         "Train model + write model card"],
        ["MRM Crew",
         fmt_time(manual_metrics.get("mrm_time_sec")),
         fmt_time(auto_metrics.get("mrm_time_sec")),
         "Compliance, stress test, verdict"],
        ["Run Time Total",
         fmt_time(manual_total),
         fmt_time(auto_total),
         "Automated wall-clock time only"],
        ["Total incl.\nHuman Setup",
         fmt_total_with_setup(manual_total),
         fmt_time(auto_total),
         "Full cost: human effort + run time"],
    ]
    t = make_table(timing_data, [3.5 * cm, 4 * cm, 3.5 * cm, 5.5 * cm], "#27ae60")
    # Highlight the "Total incl. Human Setup" row
    last_row = len(timing_data) - 1
    t.setStyle(TableStyle([
        ("FONTNAME", (0, last_row), (-1, last_row), "Helvetica-Bold"),
        ("BACKGROUND", (0, last_row), (-1, last_row), colors.HexColor("#e8f5e9")),
        ("TEXTCOLOR", (1, last_row), (1, last_row), colors.HexColor("#c53030")),  # Manual cost in red
        ("TEXTCOLOR", (2, last_row), (2, last_row), colors.HexColor("#276749")),  # AutoCrew cost in green
    ]))
    # Highlight the human setup row
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#fff3e0")),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<i>Human setup time estimate: 6 agents × 15 min (definition) + "
        "6 tasks × 20 min (instructions) + 30 min (integration) + "
        "60 min (testing/debugging) = 300 min. "
        "Assumes an experienced CrewAI developer. "
        "AutoCrew reduces this to the ~6 s Meta Agent config phase.</i>",
        styles["Caption"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4: MRM VERDICTS + CREW COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("5. MRM Audit Verdicts", styles["SectionHead"]))

    for path_label, m in [("Path A: Manual", manual_metrics),
                           ("Path B: AutoCrew", auto_metrics)]:
        verdict = str(m.get("mrm_verdict", "N/A"))
        style = styles["VerdictApproved"] if "approved" in verdict.lower() else styles["VerdictRejected"]
        story.append(Paragraph(f"<b>{path_label}:</b>  {verdict[:300]}", style))

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("6. Architecture Comparison", styles["SectionHead"]))
    story.append(make_table([
        ["Dimension", "Path A: Manual", "Path B: AutoCrew"],
        ["Agent definition", "Hardcoded in Python", "Generated by Meta Agent LLM"],
        ["Task instructions", "Hardcoded in Python", "Generated from natural language"],
        ["Tool assignment", "Manual in code", "Auto-selected from registry"],
        ["To add a new task", "Rewrite Python classes", "Change one sentence"],
        ["Configuration saved?", "No", "Yes — crew_config.json"],
        ["Crew size", "3 + 3 (fixed)", "Dynamic (LLM decides)"],
        ["Setup time", "~300 min (hardcoding)", "~6 seconds (LLM call)"],
    ], [4.5 * cm, 6 * cm, 6 * cm], "#8e44ad"))

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE: SYSTEM DIAGRAMS
    # ══════════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("7. System Diagrams", styles["SectionHead"]))

    def _embed_figure(path, caption, fig_num, w=16*cm, h=8*cm):
        if path and os.path.exists(path):
            story.append(Image(path, width=w, height=h))
            story.append(Paragraph(f"Figure {fig_num}: {caption}", styles["Caption"]))
            story.append(Spacer(1, 0.15 * inch))

    # Architecture overview
    story.append(Paragraph(
        "The diagrams below show the AutoCrew system architecture, the Meta Agent 5-step "
        "pipeline, the auto-generated crew structure, and the automation level comparison.",
        styles["BodyText2"]))
    story.append(Spacer(1, 0.1 * inch))

    _embed_figure(arch_path,
                  "AutoCrew system architecture — Meta Agent → Crew Factory → Modeling & MRM Crews",
                  fig_num=10, w=16*cm, h=9*cm)
    _embed_figure(pipeline_path,
                  "Meta Agent 5-step pipeline — each step is one LLM call producing JSON",
                  fig_num=11, w=16*cm, h=6*cm)
    story.append(PageBreak())
    _embed_figure(crew_path,
                  "Auto-generated crew structure — Modeling Crew (left) and MRM Crew (right)",
                  fig_num=12, w=16*cm, h=7*cm)
    _embed_figure(comparison_path,
                  "Automation level comparison (left) and AutoCrew execution time breakdown (right)",
                  fig_num=13, w=16*cm, h=7*cm)

    # BUILD
    try:
        doc.build(story)
        print(f"\n[Report] PDF saved to: {output_path}")
    except PermissionError:
        alt = output_path.replace(".pdf", f"_{datetime.now().strftime('%H%M%S')}.pdf")
        doc2 = SimpleDocTemplate(alt, pagesize=A4,
                                 rightMargin=1.5 * cm, leftMargin=1.5 * cm,
                                 topMargin=2 * cm, bottomMargin=2 * cm)
        doc2.build(story)
        print(f"\n[Report] PDF saved to: {alt} (original was locked)")

    return output_path


if __name__ == "__main__":
    generate_report(mode="auto")
