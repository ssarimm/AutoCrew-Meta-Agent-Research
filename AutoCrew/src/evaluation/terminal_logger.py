"""
TerminalLogger: captures all stdout/stderr during a pipeline run
and saves it as a timestamped PDF in results/.
"""
import os
import re
import sys
from datetime import datetime
from io import StringIO


# Strip ANSI escape codes (colours, cursor moves, etc.)
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfhilmnprsu]')


class _Tee:
    """Writes to both the real stream and an in-memory buffer."""

    def __init__(self, real_stream, buffer: StringIO):
        self._real = real_stream
        self._buf = buffer

    def write(self, text: str):
        self._real.write(text)
        self._buf.write(_ANSI_RE.sub('', text))

    def flush(self):
        self._real.flush()

    def fileno(self):
        return self._real.fileno()

    # Let libraries that check for isatty() behave normally
    def isatty(self):
        return False


class TerminalLogger:
    """
    Context-manager that captures every print() / logging call and,
    on exit, saves the full transcript as a PDF.

    Usage (automatic, from main.py):
        logger = TerminalLogger()
        logger.start()
        ... run pipeline ...
        logger.stop_and_save()
    """

    def __init__(self):
        self._buffer = StringIO()
        self._orig_stdout = None
        self._orig_stderr = None

    # ------------------------------------------------------------------
    def start(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _Tee(self._orig_stdout, self._buffer)
        sys.stderr = _Tee(self._orig_stderr, self._buffer)

    def stop_and_save(self, output_path: str = None) -> str:
        # Restore real streams first so any errors below are visible
        if self._orig_stdout:
            sys.stdout = self._orig_stdout
        if self._orig_stderr:
            sys.stderr = self._orig_stderr

        if output_path is None:
            os.makedirs("results", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/terminal_log_{ts}.pdf"

        self._save_pdf(output_path)
        return output_path

    # ------------------------------------------------------------------
    def _save_pdf(self, path: str):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak
            )
            from reportlab.lib.enums import TA_CENTER
        except ImportError:
            print("[TerminalLogger] reportlab not installed — cannot save PDF.")
            return

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        doc = SimpleDocTemplate(
            path, pagesize=A4,
            rightMargin=1.5 * cm, leftMargin=1.5 * cm,
            topMargin=2 * cm, bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='LogTitle',
            parent=styles['Title'],
            fontSize=16,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            name='LogMeta',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#666666"),
            spaceAfter=12,
        ))
        styles.add(ParagraphStyle(
            name='LogBody',
            parent=styles['Code'],
            fontName='Courier',
            fontSize=7,
            leading=9,
            backColor=colors.HexColor("#f5f5f5"),
            textColor=colors.HexColor("#1a1a1a"),
            borderWidth=0.3,
            borderColor=colors.HexColor("#cccccc"),
            borderPadding=5,
            spaceAfter=4,
        ))

        story = []

        # Header
        story.append(Paragraph("AutoCrew — Terminal Log", styles['LogTitle']))
        story.append(Paragraph(
            f"Captured: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['LogMeta']
        ))
        story.append(Spacer(1, 0.3 * cm))

        # Body — split into page-friendly chunks to avoid reportlab overflow
        log_text = self._buffer.getvalue()
        if not log_text.strip():
            story.append(Paragraph("(No output captured.)", styles['LogMeta']))
        else:
            # Split on newlines so we don't slice mid-line
            lines = log_text.splitlines(keepends=True)
            CHUNK = 120          # lines per Preformatted block
            for start in range(0, len(lines), CHUNK):
                block = "".join(lines[start: start + CHUNK])
                # Escape XML special characters for reportlab
                block = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Preformatted(block, styles['LogBody']))

        try:
            doc.build(story)
            print(f"[TerminalLogger] Terminal log saved → {path}")
        except PermissionError:
            ts = datetime.now().strftime("%H%M%S")
            alt = path.replace(".pdf", f"_{ts}.pdf")
            doc2 = SimpleDocTemplate(
                alt, pagesize=A4,
                rightMargin=1.5 * cm, leftMargin=1.5 * cm,
                topMargin=2 * cm, bottomMargin=2 * cm,
            )
            doc2.build(story)
            print(f"[TerminalLogger] Terminal log saved → {alt} (original was locked)")
