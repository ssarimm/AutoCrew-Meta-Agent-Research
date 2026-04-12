from crewai.tools import BaseTool
import subprocess
import sys


# Code injected at the END of any agent script that uses matplotlib.
# Auto-saves all open figures to figures/agent_plots/ labelled by their plot title.
_PLOT_SAVE_EPILOGUE = "\n".join([
    "",
    "import os as _os_ac, time as _t_ac",
    "import matplotlib.pyplot as _plt_ac",
    "_os_ac.makedirs('figures/agent_plots', exist_ok=True)",
    "for _fn in list(_plt_ac.get_fignums()):",
    "    _fig = _plt_ac.figure(_fn)",
    "    _axes = _fig.get_axes()",
    "    _raw_title = _axes[0].get_title() if _axes else ''",
    "    _safe = ''.join(c if c.isalnum() or c == '_' else '_' for c in _raw_title)[:40]",
    "    _label = _safe or ('plot_' + str(_fn))",
    "    _path = 'figures/agent_plots/' + str(int(_t_ac.time() * 1000)) + '_' + _label + '.png'",
    "    _fig.savefig(_path, dpi=120, bbox_inches='tight')",
    "    print('[Plot saved: ' + _path + ']')",
    "_plt_ac.close('all')",
    "",
])


class CodeExecutionTool(BaseTool):
    name: str = "execute_python_code"
    description: str = (
        "Executes a python script. Input should be the actual python code string to run. "
        "IMPORTANT: Each execution is a completely fresh Python process — "
        "variables, imports, and loaded data from previous calls do NOT persist. "
        "Every script must be fully self-contained: re-import all libraries and "
        "re-load the dataset from its file path at the top of every script. "
        "Example input: 'import pandas as pd\\ndf = pd.read_csv(\"data/file.csv\")\\nprint(df.shape)'"
    )

    def _run(self, code: str) -> str:
        try:
            # --- Sanitize common JSON-parsing breakage patterns ---
            # 1. print('\ntext') → JSON \n becomes real newline → unterminated string literal
            #    Fix: remove the leading \n from print strings — use a blank print() line instead.
            import re as _re
            code = _re.sub(r"print\('\\n([^']*)'", r"print('\1')", code)
            code = _re.sub(r'print\("\\n([^"]*)"', r'print("\1")', code)

            # --- Pre-flight syntax check ---
            # Catches any remaining broken code before wasting a subprocess execution.
            # Returns specific, actionable error messages so the agent fixes the right thing.
            _FIX_HINT = (
                "\n\nJSON SERIALIZATION ERRORS — your code was broken before it ran. Two causes:\n"
                "CAUSE 1 — double-quoted strings break JSON:\n"
                "  WRONG:   print(f\"Accuracy: {accuracy:.4f}\")\n"
                "  CORRECT: print('Accuracy:', round(accuracy, 4))\n"
                "  WRONG:   pd.read_csv(\"data/file.csv\")\n"
                "  CORRECT: pd.read_csv('data/file.csv')\n"
                "CAUSE 2 — \\n inside a string literal breaks JSON:\n"
                "  WRONG:   print('\\nvalue_counts:')\n"
                "  CORRECT: print('value_counts:')   # just remove the \\n\n"
                "RULE: Use ONLY single-quoted strings. No f-strings. No \\n inside strings."
            )
            try:
                compile(code, "<string>", "exec")
            except SyntaxError as _se:
                return f"Execution Error:\n  SyntaxError: {_se}{_FIX_HINT}"

            # Prepend non-interactive matplotlib backend to prevent plt.show() from blocking
            if "matplotlib" in code or "plt" in code:
                if "matplotlib.use(" not in code:
                    code = "import matplotlib\nmatplotlib.use('Agg')\n" + code
                # Append auto-save epilogue so all figures are written to figures/agent_plots/
                code = code + _PLOT_SAVE_EPILOGUE

            with open("temp_script.py", "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                [sys.executable, "temp_script.py"],
                capture_output=True,
                text=True,
                timeout=180
            )

            if result.returncode != 0:
                error_msg = result.stderr
                hints = []
                if "was never closed" in error_msg or "unexpected eof" in error_msg.lower():
                    hints.append(
                        "ROOT CAUSE: Double-quoted strings broke JSON parsing and truncated your code. "
                        "Rewrite the entire script using ONLY single quotes. "
                        "Use print('Accuracy:', round(accuracy, 4)) NOT print(f\"Accuracy: {accuracy:.4f}\"). "
                        "Use pd.read_csv('data/file.csv') NOT pd.read_csv(\"data/file.csv\")."
                    )
                if "keyerror" in error_msg.lower():
                    hints.append(
                        "Hint: The target column name does not exist in the CSV. "
                        "IMPORTANT: Do NOT trust column names from previous task outputs — prior agents may have hallucinated renamed columns (e.g. 'p' instead of '+'). "
                        "Always detect the target from the raw CSV: "
                        "known = {'SeriousDlqin2yrs', 'Class'}; target_col = next((c for c in known if c in df.columns), df.columns[-1]); "
                        "X = df.drop(columns=[target_col] + [c for c in df.columns if 'unnamed' in c.lower()]); y = df[target_col]"
                    )
                if "nan" in error_msg.lower() and ("polynomialfeatures" in error_msg.lower() or "standardscaler" in error_msg.lower() or "contains nan" in error_msg.lower()):
                    hints.append("Hint: Handle NaN BEFORE any sklearn transformer. Add: df = df.dropna()  OR  df = df.fillna(df.median(numeric_only=True))")
                if "filenotfounderror" in error_msg.lower() or "no such file or directory" in error_msg.lower():
                    hints.append("Hint: Use the exact dataset path from your task description. NEVER generate synthetic or fake data as a substitute.")
                if hints:
                    error_msg += "\n\n" + "\n".join(hints)
                return f"Execution Error:\n{error_msg}"

            output = result.stdout.strip()
            if not output:
                return "Code executed but printed nothing. Did you forget print()?"

            # Truncate very long outputs to prevent LLM context overflow
            if len(output) > 3000:
                output = output[:3000] + f"\n... [output truncated at 3000 chars — {len(result.stdout.strip())} total]"

            return f"Execution Output:\n{output}"

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (180s limit)."
        except Exception as e:
            return f"System Error: {str(e)}"


# Singleton instance
code_execution_tool = CodeExecutionTool()
