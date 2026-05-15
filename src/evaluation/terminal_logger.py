#src/evaluation/terminal_logger.py
"""
TerminalLogger: captures all stdout/stderr during a pipeline run
and saves it as a timestamped PDF in results/.
"""

import os
import re
import sys
from datetime import datetime
from io import StringIO


# ─────────────────────────────────────────────────────────────
# Remove ANSI color codes (terminal styling)
# ─────────────────────────────────────────────────────────────
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mABCDEFGHJKSTfhilmnprsu]')


class _Tee:
    """
    Duplicate stdout/stderr:
    - real terminal output
    - internal buffer (clean text)
    """

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

    def isatty(self):
        return False

    # important for compatibility with some libs
    def writable(self):
        return True

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")


class TerminalLogger:
    """
    Captures all terminal output and exports it as PDF.
    """

    def __init__(self):
        self._buffer = StringIO()
        self._orig_stdout = None
        self._orig_stderr = None
        self._active = False

    # ─────────────────────────────────────────────
    # START CAPTURE
    # ─────────────────────────────────────────────
    def start(self):
        if self._active:
            return

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        sys.stdout = _Tee(self._orig_stdout, self._buffer)
        sys.stderr = _Tee(self._orig_stderr, self._buffer)

        self._active = True

    # ─────────────────────────────────────────────
    # STOP CAPTURE
    # ─────────────────────────────────────────────
    def stop(self):
        if not self._active:
            return

        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._active = False

    # ─────────────────────────────────────────────
    # GET LOGS (NOW SAFE)
    # ─────────────────────────────────────────────
    def get_logs(self) -> str:
        self._buffer.seek(0)
        return self._buffer.getvalue()

    # ─────────────────────────────────────────────
    # STOP + SAVE PDF
    # ─────────────────────────────────────────────
    def stop_and_save(self, output_path: str = None) -> str:
        self.stop()

        if output_path is None:
            os.makedirs("results", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/terminal_log_{ts}.pdf"

        self._save_pdf(output_path)
        return output_path

    # ─────────────────────────────────────────────
    # PDF EXPORT
    # ─────────────────────────────────────────────
    def _save_pdf(self, path: str):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
            from reportlab.lib.enums import TA_CENTER
        except ImportError:
            print("[TerminalLogger] reportlab not installed.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)

        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()

        styles.add(ParagraphStyle(
            name='LogTitle',
            parent=styles['Title'],
            fontSize=16,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=8,
        ))

        styles.add(ParagraphStyle(
            name='LogMeta',
            parent=styles['Normal'],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#666"),
            spaceAfter=10,
        ))

        styles.add(ParagraphStyle(
            name='LogBody',
            parent=styles['Code'],
            fontName='Courier',
            fontSize=7,
            leading=9,
            backColor=colors.HexColor("#f5f5f5"),
            textColor=colors.HexColor("#111"),
            borderWidth=0.3,
            borderColor=colors.HexColor("#ccc"),
            borderPadding=5,
        ))

        story = []

        story.append(Paragraph("AutoCrew — Terminal Log", styles['LogTitle']))
        story.append(Paragraph(
            f"Captured: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['LogMeta']
        ))
        story.append(Spacer(1, 0.3 * cm))

        log_text = self.get_logs()

        if not log_text.strip():
            story.append(Paragraph("(No output captured.)", styles['LogMeta']))
        else:
            lines = log_text.splitlines(keepends=True)
            CHUNK = 120

            for i in range(0, len(lines), CHUNK):
                block = "".join(lines[i:i + CHUNK])

                # escape XML
                block = (
                    block.replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;")
                )

                story.append(Preformatted(block, styles['LogBody']))

        doc.build(story)
        print(f"[TerminalLogger] Saved → {path}")