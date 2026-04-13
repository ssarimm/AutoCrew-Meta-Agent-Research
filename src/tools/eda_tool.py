from crewai.tools import BaseTool
import subprocess
import sys


class EDAtool(BaseTool):
    name: str = "exploratory_data_analysis"
    description: str = (
        "Runs exploratory data analysis on a CSV dataset. "
        "Input should be the path to a CSV file. "
        "Returns shape, dtypes, missing values, class balance, and basic statistics."
    )

    def _run(self, data_path: str) -> str:
        # Detect if file needs header=None
        header_arg = ""
        if "credit_card_approval" in data_path:
            header_arg = ", header=None"

        eda_code = f'''
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("{data_path}"{header_arg})

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS REPORT")
print("=" * 50)

print(f"\\nShape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
print(f"\\nColumns: {{list(df.columns)}}")

print("\\n--- Data Types ---")
print(df.dtypes.to_string())

print("\\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({{"count": missing, "percent": missing_pct}})
missing_df = missing_df[missing_df["count"] > 0]
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("No missing values found.")

# Also check for '?' placeholder values
q_count = (df == '?').sum().sum() if df.dtypes.apply(lambda x: x == object).any() else 0
if q_count > 0:
    print(f"\\nWARNING: Found {{q_count}} '?' placeholder values that should be treated as NaN.")

print("\\n--- Descriptive Statistics ---")
print(df.describe().round(3).to_string())

print("\\n--- Class Balance (last column) ---")
target = df.columns[-1]
vc = df[target].value_counts()
print(f"Target column: {{target}}")
for val, count in vc.items():
    pct = round(count / len(df) * 100, 2)
    print(f"  {{val}}: {{count}} ({{pct}}%)")

print("\\n--- Correlation (top pairs) ---")
numeric_df = df.select_dtypes(include=["number"])
if len(numeric_df.columns) > 1:
    corr = numeric_df.corr().abs()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if val > 0.5:
                pairs.append((cols[i], cols[j], round(val, 3)))
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
    if pairs:
        for a, b, c in pairs:
            print(f"  {{a}} <-> {{b}}: {{c}}")
    else:
        print("  No strong correlations (>0.5) found.")
else:
    print("  Not enough numeric columns.")
'''
        try:
            with open("temp_eda.py", "w", encoding="utf-8") as f:
                f.write(eda_code)

            result = subprocess.run(
                [sys.executable, "temp_eda.py"],
                capture_output=True,
                text=True,
                timeout=120
            )

            # Cleanup temp file
            import os
            try:
                if os.path.exists("temp_eda.py"):
                    os.remove("temp_eda.py")
            except OSError:
                pass

            if result.returncode != 0:
                return f"EDA Error:\n{result.stderr}"

            output = result.stdout.strip()
            if len(output) > 3000:
                output = output[:3000] + f"\n... [EDA output truncated at 3000 chars — {len(result.stdout.strip())} total]"
            return output

        except Exception as e:
            return f"EDA System Error: {str(e)}"


# Singleton
eda_tool = EDAtool()
