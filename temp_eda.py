
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/credit_card_approval.csv")

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS REPORT")
print("=" * 50)

print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

print("\n--- Data Types ---")
print(df.dtypes.to_string())

print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
missing_df = missing_df[missing_df["count"] > 0]
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("No missing values found.")

print("\n--- Descriptive Statistics ---")
print(df.describe().round(3).to_string())

print("\n--- Class Balance (last column) ---")
target = df.columns[-1]
vc = df[target].value_counts()
print(f"Target column: {target}")
for val, count in vc.items():
    pct = round(count / len(df) * 100, 2)
    print(f"  {val}: {count} ({pct}%)")

print("\n--- Correlation (top pairs) ---")
numeric_df = df.select_dtypes(include=["number"])
if len(numeric_df.columns) > 1:
    corr = numeric_df.corr().abs()
    upper = corr.where(
        pd.np.triu(pd.np.ones(corr.shape), k=1).astype(bool)
    ) if hasattr(pd, "np") else corr
    pairs = []
    for col in corr.columns:
        for idx in corr.index:
            if col != idx and abs(corr.loc[idx, col]) > 0.5:
                pairs.append((idx, col, round(corr.loc[idx, col], 3)))
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
    if pairs:
        for a, b, c in pairs:
            print(f"  {a} <-> {b}: {c}")
    else:
        print("  No strong correlations (>0.5) found.")
else:
    print("  Not enough numeric columns.")
