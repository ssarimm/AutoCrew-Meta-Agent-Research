"""
AutoCrew PDF Report Generator
Generates a professional PDF report for FYP presentation.
Images are read from the figures/ folder.
Run: python -m src.report_generator
"""
import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, Preformatted
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Path to figures folder (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


def generate_report(
    mode="auto",
    config_path=None,
    results=None,
    output_path="results/AutoCrew_Report.pdf"
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate charts if figures/ folder is empty or missing
    if not os.path.exists(FIGURES_DIR) or len(os.listdir(FIGURES_DIR)) < 3:
        print("[Report] Generating charts into figures/ ...")
        try:
            import sys
            sys.path.insert(0, PROJECT_ROOT)
            from generate_charts import generate_all
            generate_all()
        except Exception as e:
            print(f"[Report] Chart generation failed: {e}")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='MainTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        name='SubTitle', parent=styles['Normal'],
        fontSize=12, alignment=TA_CENTER,
        textColor=colors.HexColor("#666666"), spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        name='SectionHead', parent=styles['Heading1'],
        fontSize=16, textColor=colors.HexColor("#16213e"),
        spaceBefore=20, spaceAfter=10,
        borderWidth=1, borderColor=colors.HexColor("#0f3460"), borderPadding=5,
    ))
    styles.add(ParagraphStyle(
        name='SubSection', parent=styles['Heading2'],
        fontSize=13, textColor=colors.HexColor("#0f3460"),
        spaceBefore=14, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name='BodyText2', parent=styles['Normal'],
        fontSize=10, leading=14, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name='CodeStyle', parent=styles['Code'],
        fontSize=7.5, leading=10,
        backColor=colors.HexColor("#f4f4f4"),
        borderWidth=0.5, borderColor=colors.HexColor("#cccccc"), borderPadding=6,
    ))
    styles.add(ParagraphStyle(
        name='CaptionStyle', parent=styles['Normal'],
        fontSize=9, alignment=TA_CENTER,
        textColor=colors.HexColor("#888888"), spaceAfter=12,
    ))

    story = []

    def add_figure(filename, width_cm=16, height_cm=10, caption=None):
        """Helper to add a figure from figures/ folder."""
        path = os.path.join(FIGURES_DIR, filename)
        if os.path.exists(path):
            story.append(Spacer(1, 0.15*inch))
            story.append(Image(path, width=width_cm*cm, height=height_cm*cm))
            if caption:
                story.append(Paragraph(caption, styles['CaptionStyle']))
            return True
        else:
            story.append(Paragraph(
                f"<i>[Figure not found: {filename}. Run: python -m src.generate_charts]</i>",
                styles['BodyText2']
            ))
            return False

    def make_table(data, widths, header_color="#16213e"):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ]))
        return t

    # ==================== COVER PAGE ====================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("AutoCrew", styles['MainTitle']))
    story.append(Paragraph(
        "Automatic Generation of Agentic AI Crews<br/>for Financial Modeling Tasks",
        styles['SubTitle']
    ))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        f"Execution Report &mdash; {datetime.now().strftime('%B %d, %Y')}", styles['SubTitle']
    ))
    story.append(Paragraph(
        f"Mode: <b>{'AutoCrew (Meta Agent)' if mode == 'auto' else 'Manual Pipeline'}</b>",
        styles['SubTitle']
    ))
    story.append(Spacer(1, 0.5*inch))

    info_data = [
        ["Project", "AutoCrew - Final Year Project (BSCS)"],
        ["University", "FAST National University, Karachi"],
        ["Department", "Department of Computer Science"],
        ["Base Paper", "Agentic AI for Model Development & MRM (Okpala et al., 2025)"],
        ["Framework", "CrewAI + LangChain + LiteLLM"],
        ["LLM Provider", "Groq (Llama 3.3 70B) / Google Gemini / Ollama"],
        ["Dataset", results.get("dataset", "data/credit_card_approval.csv") if results else "data/credit_card_approval.csv"],
    ]
    it = Table(info_data, colWidths=[3.5*cm, 13*cm])
    it.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor("#0f3460")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, -2), 0.3, colors.HexColor("#e0e0e0")),
    ]))
    story.append(it)
    story.append(PageBreak())

    # ==================== TOC ====================
    story.append(Paragraph("Table of Contents", styles['SectionHead']))
    toc = [
        "1. Project Overview",
        "2. System Architecture",
        "3. Meta Agent Pipeline",
        "4. Generated Crew Configuration (JSON)",
        "5. Crew Structure Visualization",
        "6. EDA Visualizations",
        "7. Execution Results",
        "8. Manual vs AutoCrew Comparison",
        "9. Code Structure",
        "10. Conclusion & Future Work",
    ]
    for item in toc:
        story.append(Paragraph(item, styles['BodyText2']))
    story.append(PageBreak())

    # ==================== 1. OVERVIEW ====================
    story.append(Paragraph("1. Project Overview", styles['SectionHead']))
    story.append(Paragraph(
        "AutoCrew is an R&amp;D extension of the paper "
        "<i>\"Agentic AI Systems for Financial Services\"</i> by Okpala et al. (arXiv:2502.05439, 2025). "
        "The base paper manually codes AI agent crews. AutoCrew introduces a <b>Meta Agent</b> that "
        "automatically generates crew configurations from natural language, eliminating hardcoded definitions.",
        styles['BodyText2']
    ))
    story.append(Paragraph(
        "The system takes a task like <i>\"Predict credit approval\"</i> and autonomously: "
        "(1) decomposes it into subtasks, (2) selects agent roles, (3) assigns tools, "
        "(4) writes instructions, and (5) generates a complete JSON configuration.",
        styles['BodyText2']
    ))

    story.append(Paragraph("1.1 Key Contributions", styles['SubSection']))
    story.append(make_table([
        ["#", "Contribution", "Description"],
        ["1", "Meta Agent", "LLM-powered 5-step orchestrator for crew creation"],
        ["2", "JSON Config", "Structured crew_config.json from NL input"],
        ["3", "Crew Factory", "JSON parser that builds live CrewAI objects"],
        ["4", "Dual-Path", "Manual vs Auto for fair comparison"],
        ["5", "Tool Registry", "Extensible string-to-tool mapping"],
        ["6", "Evaluation", "Side-by-side metrics (accuracy, F1, timing)"],
    ], [0.8*cm, 2.5*cm, 13.7*cm]))
    story.append(PageBreak())

    # ==================== 2. ARCHITECTURE ====================
    story.append(Paragraph("2. System Architecture", styles['SectionHead']))
    story.append(Paragraph(
        "AutoCrew uses a dual-path architecture. <b>Path A</b> (Manual) preserves the base paper. "
        "<b>Path B</b> (AutoCrew) uses the Meta Agent. Both share tools, HITL gate, and CrewAI engine.",
        styles['BodyText2']
    ))
    add_figure("architecture.png", 16, 9.3, "Figure 1: AutoCrew dual-path system architecture")

    story.append(make_table([
        ["Component", "Path A (Manual)", "Path B (AutoCrew)"],
        ["Agent Definition", "Hardcoded in Python", "Generated by LLM"],
        ["Task Definition", "Hardcoded in Python", "Generated from NL"],
        ["Tool Assignment", "Manual in code", "LLM selects from registry"],
        ["Configuration", "Python classes", "JSON config file"],
        ["Adaptability", "Requires code changes", "Just change the prompt"],
        ["Execution", "CrewAI", "CrewAI (identical)"],
    ], [4*cm, 5.5*cm, 7.5*cm], "#0f3460"))
    story.append(PageBreak())

    # ==================== 3. PIPELINE ====================
    story.append(Paragraph("3. Meta Agent Pipeline", styles['SectionHead']))
    story.append(Paragraph(
        "The Meta Agent operates as a <b>5-step pipeline</b>. Each step makes a structured LLM call "
        "with JSON output. If LLM fails, a rule-based fallback produces valid configuration.",
        styles['BodyText2']
    ))
    add_figure("pipeline.png", 16, 4.7, "Figure 2: Meta Agent 5-step pipeline")

    story.append(make_table([
        ["Step", "Module", "Input", "Output", "Time"],
        ["1", "Task Decomposer", "NL task description", "Subtask list", "~1s"],
        ["2", "Agent Selector", "Subtask list", "Role + persona", "~1s"],
        ["3", "Tool Assigner", "Agents + catalog", "Tool names", "~1s"],
        ["4", "Instruction Writer", "Full context", "Descriptions", "~2s"],
        ["5", "JSON Generator", "All above", "crew_config.json", "<1s"],
    ], [0.8*cm, 2.8*cm, 3.5*cm, 3.5*cm, 1.2*cm], "#533483"))
    story.append(Paragraph(
        "<b>Total: ~6 seconds</b> on Groq. This replaces hours of manual Python coding.",
        styles['BodyText2']
    ))
    story.append(PageBreak())

    # ==================== 4. JSON CONFIG ====================
    story.append(Paragraph("4. Generated JSON Configuration", styles['SectionHead']))

    config_data = None
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
    else:
        config_dir = os.path.join(PROJECT_ROOT, "configs", "generated")
        if os.path.exists(config_dir):
            files = sorted([f for f in os.listdir(config_dir) if f.endswith(".json")])
            if files:
                cp = os.path.join(config_dir, files[-1])
                with open(cp, "r") as f:
                    config_data = json.load(f)
                story.append(Paragraph(f"Config: <font color='blue'>{cp}</font>", styles['BodyText2']))

    if config_data:
        json_text = json.dumps(config_data, indent=2, ensure_ascii=False)
        if len(json_text) > 4000:
            json_text = json_text[:4000] + "\n... [truncated]"
        story.append(Preformatted(json_text, styles['CodeStyle']))

        story.append(PageBreak())
        story.append(Paragraph("4.1 Generated Agents Summary", styles['SubSection']))
        for crew_name in ["modeling_crew", "mrm_crew"]:
            crew = config_data.get(crew_name, {})
            agents = crew.get("agents", [])
            if agents:
                label = crew_name.replace("_", " ").title()
                story.append(Paragraph(f"<b>{label}</b>", styles['BodyText2']))
                rows = [["#", "Role", "Goal", "Tools"]]
                for i, a in enumerate(agents):
                    tools = ", ".join(a.get("tools", [])) or "None"
                    rows.append([str(i+1), a.get("role", ""), a.get("goal", "")[:60], tools])
                story.append(make_table(rows, [0.8*cm, 3.5*cm, 7.5*cm, 5.2*cm], "#2c3e50"))
                story.append(Spacer(1, 0.2*inch))
    else:
        story.append(Paragraph("No config found. Run the pipeline first.", styles['BodyText2']))
    story.append(PageBreak())

    # ==================== 5. CREW STRUCTURE ====================
    story.append(Paragraph("5. Crew Structure Visualization", styles['SectionHead']))
    story.append(Paragraph(
        "The Meta Agent automatically determined the crew composition. "
        "Each agent was assigned tools from the centralized tool registry.",
        styles['BodyText2']
    ))
    add_figure("crew_structure.png", 16, 5.8, "Figure 3: Auto-generated crew structure with tool assignments")

    story.append(make_table([
        ["Tool", "Purpose", "Used By"],
        ["code_execution", "Runs Python scripts (pandas, sklearn)", "Analyst, Sr.DS, ML Eng, Stress Tester"],
        ["eda_tool", "Automated EDA with labeled plots", "Data Scientist"],
        ["cag_tool", "Compliance checking vs guidelines", "Compliance Officer"],
    ], [3*cm, 7*cm, 7*cm]))
    story.append(PageBreak())

    # ==================== 6. EDA ====================
    story.append(Paragraph("6. EDA Visualizations", styles['SectionHead']))
    story.append(Paragraph(
        "The EDA tool generates labeled plots for all dataset columns. "
        "The credit card approval dataset has <b>690 rows x 16 columns</b> with "
        "class balance: 55.6% negative vs 44.4% positive.",
        styles['BodyText2']
    ))
    add_figure("eda_plots.png", 16, 9.5, "Figure 4: EDA plots with distributions, value counts, and class balance")

    story.append(make_table([
        ["Statistic", "Col '0'", "Col '1.25'", "Col '01'", "Col '0.1'"],
        ["Mean", "4.766", "2.225", "2.402", "1018.9"],
        ["Std Dev", "4.978", "3.349", "4.866", "5213.7"],
        ["Min", "0.000", "0.000", "0", "0"],
        ["Max", "28.000", "28.500", "67", "100,000"],
    ], [2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm]))
    story.append(PageBreak())

    # ==================== 7. RESULTS ====================
    story.append(Paragraph("7. Execution Results", styles['SectionHead']))

    if results:
        story.append(Paragraph(f"<b>Mode:</b> {results.get('mode', 'N/A')}", styles['BodyText2']))
        story.append(Paragraph(f"<b>Dataset:</b> {results.get('dataset', 'N/A')}", styles['BodyText2']))
        story.append(Paragraph(f"<b>Task:</b> {results.get('task', 'N/A')}", styles['BodyText2']))

        timing = [["Phase", "Time (seconds)"]]
        for key, label in [("meta_agent_time_sec", "Meta Agent"), ("modeling_time_sec", "Modeling"), ("mrm_time_sec", "MRM"), ("total_time_sec", "Total")]:
            if results.get(key):
                timing.append([label, str(results[key])])
        if len(timing) > 1:
            story.append(make_table(timing, [6*cm, 5*cm], "#27ae60"))

        verdict = results.get("mrm_verdict", "")
        if verdict and "rate_limit" not in verdict.lower():
            story.append(Paragraph("7.1 MRM Verdict", styles['SubSection']))
            story.append(Preformatted(str(verdict)[:2000], styles['CodeStyle']))
    else:
        story.append(Paragraph("No results available. Run the pipeline first.", styles['BodyText2']))
    story.append(PageBreak())

    # ==================== 8. COMPARISON ====================
    story.append(Paragraph("8. Manual vs AutoCrew Comparison", styles['SectionHead']))
    add_figure("comparison.png", 16, 6.6, "Figure 5: Automation level and execution time comparison")

    story.append(make_table([
        ["Dimension", "Manual (Base Paper)", "AutoCrew (Meta Agent)"],
        ["Setup Method", "Write Python classes", "Provide NL description"],
        ["Time to Configure", "Hours (per task)", "~6 seconds"],
        ["Agent Count", "3+3 = 6 (fixed)", "Dynamic (LLM decides)"],
        ["Tool Assignment", "Hardcoded", "Auto from registry"],
        ["New Task", "Rewrite Python", "Change one sentence"],
        ["JSON Config", "Not generated", "Saved to file"],
        ["Developer Skill", "Python + CrewAI", "Describe task in English"],
    ], [3.5*cm, 6.5*cm, 7*cm], "#8e44ad"))
    story.append(PageBreak())

    # ==================== 9. CODE STRUCTURE ====================
    story.append(Paragraph("9. Code Structure", styles['SectionHead']))
    files = [
        "AutoCrew/",
        "  src/main.py .................. Entry point (manual/auto/compare)",
        "  src/config.py ................ LLM provider setup",
        "  src/meta_agent/",
        "    orchestrator.py ............ Chains 5 pipeline steps",
        "    task_decomposer.py ......... NL -> subtasks",
        "    agent_selector.py .......... Subtask -> agent",
        "    tool_assigner.py ........... Agent -> tools",
        "    instruction_writer.py ...... Generate instructions",
        "    json_generator.py .......... Assemble JSON",
        "  src/crew_factory/",
        "    crew_builder.py ............ JSON -> live CrewAI",
        "    auto_runner.py ............. Auto pipeline runner",
        "  src/manual/ .................. Base paper (Path A)",
        "  src/tools/ ................... Shared tools",
        "  src/evaluation/ .............. Comparison framework",
        "  src/generate_charts.py ....... Chart generator",
        "  src/report_generator.py ...... This PDF generator",
        "  figures/ ..................... Generated charts (PNG)",
        "  configs/generated/ ........... Auto-saved JSON configs",
    ]
    for line in files:
        story.append(Preformatted(line, styles['CodeStyle']))
    story.append(PageBreak())

    # ==================== 10. CONCLUSION ====================
    story.append(Paragraph("10. Conclusion & Future Work", styles['SectionHead']))
    story.append(Paragraph(
        "AutoCrew demonstrates that a Meta Agent can automate multi-agent crew creation "
        "for financial modeling. Key achievements:", styles['BodyText2']
    ))
    for c in [
        "Reduced developer effort: no code changes for new tasks",
        "Dynamic crew sizing: Meta Agent decides agent count",
        "Structured output: JSON configs are reusable and versionable",
        "Graceful degradation: rule-based fallbacks when LLM fails",
        "Built-in evaluation: side-by-side comparison framework",
    ]:
        story.append(Paragraph(f"&bull; {c}", styles['BodyText2']))

    story.append(Paragraph("<b>Future Work:</b>", styles['SubSection']))
    for f in [
        "Hierarchical crew structures (manager delegation)",
        "Vector DB integration for CAG (semantic compliance)",
        "Web UI for non-technical users",
        "More LLM providers (Claude, GPT-4)",
        "Agent memory across runs",
        "MLflow integration for experiment tracking",
    ]:
        story.append(Paragraph(f"&bull; {f}", styles['BodyText2']))

    # BUILD
    try:
        doc.build(story)
        print(f"\n[Report] PDF saved to: {output_path}")
    except PermissionError:
        alt_path = f"results/AutoCrew_Report_{datetime.now().strftime('%H%M%S')}.pdf"
        doc2 = SimpleDocTemplate(alt_path, pagesize=A4,
                                  rightMargin=1.5*cm, leftMargin=1.5*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
        doc2.build(story)
        print(f"\n[Report] PDF saved to: {alt_path} (original was locked)")

    return output_path


if __name__ == "__main__":
    config_dir = os.path.join(PROJECT_ROOT, "configs", "generated")
    config_path = None
    if os.path.exists(config_dir):
        files = sorted([f for f in os.listdir(config_dir) if f.endswith(".json")])
        if files:
            config_path = os.path.join(config_dir, files[-1])

    generate_report(mode="auto", config_path=config_path)
