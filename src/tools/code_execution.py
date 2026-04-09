from crewai.tools import BaseTool
import subprocess
import sys


class CodeExecutionTool(BaseTool):
    name: str = "execute_python_code"
    description: str = (
        "Executes a python script. Input should be the actual python code string to run. "
        "Example input: 'print(1+1)'."
    )

    def _run(self, code: str) -> str:
        try:
            with open("temp_script.py", "w", encoding="utf-8") as f:
                f.write(code)

            result = subprocess.run(
                [sys.executable, "temp_script.py"],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return f"Execution Error:\n{result.stderr}"

            output = result.stdout.strip()
            if not output:
                return "Code executed but printed nothing. Did you forget print()?"

            return f"Execution Output:\n{output}"

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (120s limit)."
        except Exception as e:
            return f"System Error: {str(e)}"


# Singleton instance
code_execution_tool = CodeExecutionTool()
