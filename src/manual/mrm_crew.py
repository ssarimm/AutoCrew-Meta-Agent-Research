from crewai import Agent, Task, Crew, Process
from src.tools.code_execution import code_execution_tool


def _build_stress_test_code(dataset_path):
    """
    Build dataset-specific stress test code.
    Returns (load_block, rf_params, sampling_block).
    """
    if "credit_card_approval" in dataset_path:
        load_block = (
            f"df = pd.read_csv('{dataset_path}', header=None)\n"
            f"df = df.replace('?', np.nan)\n"
            f"df = df.dropna()\n"
            f"for col in df.select_dtypes(include='object').columns:\n"
            f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
            f"target_col = df.columns[-1]\n"
            f"X = df.drop(columns=[target_col])\n"
            f"y = df[target_col]\n"
        )
        rf_params = "n_estimators=100, random_state=42"
        sampling_block = ""

    elif "cs-training" in dataset_path:
        load_block = (
            f"df = pd.read_csv('{dataset_path}')\n"
            f"df = df.drop(columns=[c for c in df.columns if 'unnamed' in c.lower()], errors='ignore')\n"
            f"df = df.fillna(df.median(numeric_only=True))\n"
            f"for col in df.select_dtypes(include='object').columns:\n"
            f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
            f"target_col = 'SeriousDlqin2yrs'\n"
            f"X = df.drop(columns=[target_col])\n"
            f"y = df[target_col]\n"
        )
        rf_params = "n_estimators=100, random_state=42"
        sampling_block = ""

    elif "creditcard_2023" in dataset_path:
        load_block = (
            f"df = pd.read_csv('{dataset_path}')\n"
            f"df = df.drop(columns=[c for c in df.columns if c.lower() in ('id', 'unnamed: 0')], errors='ignore')\n"
            f"df = df.dropna()\n"
            f"for col in df.select_dtypes(include='object').columns:\n"
            f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
            f"target_col = 'Class'\n"
            f"X = df.drop(columns=[target_col])\n"
            f"y = df[target_col]\n"
        )
        rf_params = "n_estimators=20, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced'"
        sampling_block = (
            f"# Downsample majority class for speed\n"
            f"if len(X_train) > 100000:\n"
            f"    import pandas as _pd_tmp\n"
            f"    _combined = _pd_tmp.concat([X_train, y_train], axis=1)\n"
            f"    _majority = _combined[_combined[target_col] == 0]\n"
            f"    _minority = _combined[_combined[target_col] == 1]\n"
            f"    _majority_down = _majority.sample(n=min(50000, len(_majority)), random_state=42)\n"
            f"    _combined = _pd_tmp.concat([_majority_down, _minority])\n"
            f"    X_train = _combined.drop(columns=[target_col])\n"
            f"    y_train = _combined[target_col]\n"
            f"    print('Downsampled training set:', X_train.shape)\n"
        )

    else:
        load_block = (
            f"df = pd.read_csv('{dataset_path}')\n"
            f"df = df.replace('?', np.nan)\n"
            f"df = df.dropna()\n"
            f"for col in df.select_dtypes(include='object').columns:\n"
            f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
            f"known = {{'SeriousDlqin2yrs', 'Class'}}\n"
            f"target_col = next((c for c in known if c in df.columns), df.columns[-1])\n"
            f"X = df.drop(columns=[target_col] + [c for c in df.columns if 'unnamed' in c.lower()])\n"
            f"y = df[target_col]\n"
        )
        rf_params = "n_estimators=100, random_state=42"
        sampling_block = ""

    return load_block, rf_params, sampling_block


class ManualMRMCrew:
    """
    PATH A: Hardcoded MRM Crew from the base paper.
    """

    def __init__(self, llm):
        self.llm = llm

    def run(self, modeling_output, dataset_path="data/credit_card_approval.csv"):
        load_block, rf_params, sampling_block = _build_stress_test_code(dataset_path)

        # === AGENTS ===

        compliance = Agent(
            role="Compliance Officer",
            goal="Check if documentation is complete.",
            backstory="You verify if Accuracy and F1 Score metrics were reported in the Model Card.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

        stress_tester = Agent(
            role="Stress Testing Engineer",
            goal="Verify model robustness by stress testing on the real dataset.",
            backstory=(
                "You write complete, self-contained Python scripts to train a model and stress test it. "
                "You ALWAYS load the real dataset from the given path. "
                "You NEVER generate synthetic or fake data. "
                "You NEVER fabricate or hallucinate results — if code times out, report the error honestly."
            ),
            tools=[code_execution_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=8,
        )

        judge = Agent(
            role="Chief Risk Officer",
            goal="Issue final APPROVED or REJECTED verdict based on compliance and stress test.",
            backstory="You review compliance and stress test results and issue a clear verdict.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
        )

        # === TASKS ===

        task1 = Task(
            description=(
                f"Review this Model Card and check if it is complete:\n\n"
                f"{str(modeling_output)[:2000]}\n\n"
                f"Check for: Accuracy value, F1 Score value, algorithm name, dataset used. "
                f"Output: PASS if all present, FAIL with missing items listed."
            ),
            expected_output="Compliance Check: PASS or FAIL with list of missing items.",
            agent=compliance,
        )

        task2 = Task(
            description=(
                f"Write and execute a complete Python script to stress test the model. "
                f"Load the REAL dataset from '{dataset_path}' — NEVER generate fake data. "
                f"NEVER fabricate or invent metric values — only report numbers printed by actual code execution. "
                f"Complete ALL steps in ONE single script:\n"
                f"Step 1: import pandas as pd, numpy as np\n"
                f"Step 2: from sklearn.model_selection import train_test_split\n"
                f"Step 3: from sklearn.ensemble import RandomForestClassifier\n"
                f"Step 4: from sklearn.preprocessing import LabelEncoder\n"
                f"Step 5: from sklearn.metrics import accuracy_score, f1_score\n"
                f"Step 6-8: Load and preprocess data:\n"
                f"{load_block}"
                f"Step 9: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                f"{sampling_block}"
                f"Step 10: model = RandomForestClassifier({rf_params}); model.fit(X_train, y_train)\n"
                f"Step 11: y_pred = model.predict(X_test)\n"
                f"Step 12: baseline_acc = accuracy_score(y_test, y_pred)\n"
                f"         baseline_f1 = f1_score(y_test, y_pred, average='weighted')\n"
                f"         print('Baseline Accuracy:', round(baseline_acc, 4))\n"
                f"         print('Baseline F1 Score:', round(baseline_f1, 4))\n"
                f"Step 13: STRESS TEST — multiply X_test numeric features by 1.5:\n"
                f"         X_test_stressed = X_test.copy()\n"
                f"         for col in X_test_stressed.select_dtypes(include=[np.number]).columns:\n"
                f"             X_test_stressed[col] = X_test_stressed[col] * 1.5\n"
                f"Step 14: y_pred_stressed = model.predict(X_test_stressed)\n"
                f"         stressed_acc = accuracy_score(y_test, y_pred_stressed)\n"
                f"         print('Stressed Accuracy:', round(stressed_acc, 4))\n"
                f"Step 15: if stressed_acc > 0.5:\n"
                f"             print('Stress Test Passed')\n"
                f"         else:\n"
                f"             print('Stress Test Warning: accuracy dropped below 0.5')"
            ),
            expected_output=(
                "Baseline Accuracy: X.XXXX\nBaseline F1 Score: X.XXXX\n"
                "Stressed Accuracy: X.XXXX\nStress Test Passed (or Warning)"
            ),
            agent=stress_tester,
        )

        task3 = Task(
            description=(
                "Based on the stress test results (task 2), issue a FINAL VERDICT.\n"
                "The verdict is determined ONLY by model performance — NOT by documentation completeness.\n"
                "Rules:\n"
                "- If Baseline Accuracy > 0.7 AND 'Stress Test Passed' appeared in results: APPROVED\n"
                "- If Baseline Accuracy <= 0.7 OR 'Stress Test Warning' appeared: REJECTED\n"
                "- If the stress test FAILED TO RUN (timeout or error), verdict is: INCONCLUSIVE — stress test did not complete\n"
                "Note: Compliance findings are informational only. A minor documentation issue does NOT cause rejection.\n"
                "State your verdict clearly as 'FINAL VERDICT: APPROVED' or 'FINAL VERDICT: REJECTED' or 'FINAL VERDICT: INCONCLUSIVE' "
                "followed by 1-2 sentences of reasoning based on the accuracy and stress test."
            ),
            expected_output="FINAL VERDICT: APPROVED or REJECTED or INCONCLUSIVE with reasoning.",
            agent=judge,
            context=[task1, task2],
        )

        # === CREW ===

        crew = Crew(
            agents=[compliance, stress_tester, judge],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            memory=False,
            verbose=False,
        )

        return crew.kickoff()
