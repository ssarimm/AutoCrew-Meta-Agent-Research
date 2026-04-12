from crewai import Agent, Task, Crew, Process
from src.tools.code_execution import code_execution_tool


class ManualMRMCrew:
    """
    PATH A: Hardcoded MRM Crew from the base paper.
    Agents, tasks, and tools are manually defined in Python code.
    This is kept unchanged for comparison with AutoCrew.
    """

    def __init__(self, llm):
        self.llm = llm

    def run(self, modeling_output, dataset_path="data/credit_card_approval.csv"):
        # Determine target column and columns to drop based on dataset
        if "cs-training" in dataset_path:
            target_col = "SeriousDlqin2yrs"
            drop_index = "drop any column with 'unnamed' in its name (e.g. 'Unnamed: 0')"
        elif "creditcard_2023" in dataset_path:
            target_col = "Class"
            drop_index = "no index column to drop"
        else:
            # credit_card_approval.csv — detect last column dynamically
            # (the CSV has no header row, so pandas uses the first data row as column names)
            import pandas as _pd
            try:
                _df_head = _pd.read_csv(dataset_path, nrows=0)
                target_col = _df_head.columns[-1]
            except Exception:
                target_col = "+"  # fallback: known last column when no header
            drop_index = "no index column to drop"

        # === AGENTS (hardcoded) ===

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
                "You NEVER generate synthetic or fake data."
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

        # === TASKS (hardcoded) ===

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
                f"Complete ALL steps in ONE single script:\n"
                f"Step 1: import pandas as pd, numpy as np\n"
                f"Step 2: from sklearn.model_selection import train_test_split\n"
                f"Step 3: from sklearn.ensemble import RandomForestClassifier\n"
                f"Step 4: from sklearn.preprocessing import LabelEncoder\n"
                f"Step 5: from sklearn.metrics import accuracy_score, f1_score\n"
                f"Step 6: df = pd.read_csv('{dataset_path}')\n"
                f"Step 7: df = df.replace('?', np.nan); df = df.dropna()\n"
                f"Step 8: encode all object columns: "
                f"for col in df.select_dtypes(include='object').columns: df[col] = LabelEncoder().fit_transform(df[col])\n"
                f"Step 9: {drop_index}\n"
                f"Step 10: target_col = '{target_col}'; "
                f"X = df.drop(columns=[c for c in df.columns if c == target_col or 'unnamed' in c.lower()]); "
                f"y = df[target_col]\n"
                f"Step 11: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                f"Step 12: model = RandomForestClassifier(n_estimators=100, random_state=42); model.fit(X_train, y_train)\n"
                f"Step 13: y_pred = model.predict(X_test)\n"
                f"Step 14: baseline_acc = accuracy_score(y_test, y_pred)\n"
                f"         baseline_f1 = f1_score(y_test, y_pred, average='weighted')\n"
                f"         print(f'Baseline Accuracy: {{baseline_acc:.4f}}')\n"
                f"         print(f'Baseline F1 Score: {{baseline_f1:.4f}}')\n"
                f"Step 15: STRESS TEST — multiply X_test numeric features by 1.5:\n"
                f"         X_test_stressed = X_test.copy()\n"
                f"         for col in X_test_stressed.select_dtypes(include=[np.number]).columns:\n"
                f"             X_test_stressed[col] = X_test_stressed[col] * 1.5\n"
                f"Step 16: y_pred_stressed = model.predict(X_test_stressed)\n"
                f"         stressed_acc = accuracy_score(y_test, y_pred_stressed)\n"
                f"         print(f'Stressed Accuracy: {{stressed_acc:.4f}}')\n"
                f"Step 17: if stressed_acc > 0.5:\n"
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
                "Note: Compliance findings are informational only. A minor documentation issue does NOT cause rejection.\n"
                "State your verdict clearly as 'FINAL VERDICT: APPROVED' or 'FINAL VERDICT: REJECTED' "
                "followed by 1-2 sentences of reasoning based on the accuracy and stress test."
            ),
            expected_output="FINAL VERDICT: APPROVED or REJECTED with reasoning based on accuracy and stress test.",
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
