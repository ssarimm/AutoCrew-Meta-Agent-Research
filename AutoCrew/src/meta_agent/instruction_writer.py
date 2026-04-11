import json


INSTRUCTION_PROMPT = """You are an expert at writing precise instructions for AI agents in financial modeling crews.

Given a subtask and its assigned agent role, write the detailed CrewAI task configuration.

For each subtask, produce:
1. task_description: A detailed, specific instruction for the agent. Include the dataset path if relevant. 
   Be explicit about what code to write, what to print, what format to output.
2. expected_output: What the task output should look like (1-2 sentences).

Rules:
- For code-executing agents: tell them EXACTLY what to code step by step, what libraries to import, what to print
- CRITICAL: For data loading, ALWAYS include: replace '?' with NaN, drop NaN rows, use LabelEncoder on object columns
- CRITICAL: Before using ANY sklearn transformer (StandardScaler, PolynomialFeatures, etc.), ALWAYS handle NaN first: df.dropna() or df.fillna(df.median(numeric_only=True)). Never pass NaN values to sklearn.
- CRITICAL: For model training, specify: use RandomForestClassifier(n_estimators=100, random_state=42), print Accuracy and F1 Score using sklearn.metrics
- CRITICAL: Tell agents to NEVER generate synthetic or fake data — not even as a fallback if the file is not found. Always use the exact dataset path provided. If the file is missing, the agent must report the error and stop.
- CRITICAL: Always embed the exact dataset path (e.g. 'data/cs-training.csv') directly in the task_description string. Never write vague instructions like "load the data".
- For analysis agents: tell them what to look for and how to format findings
- For writers: tell them what sections to include in the document
- ALWAYS include the dataset path in descriptions where the agent needs to load data
- Be specific about metric names: "Accuracy", "F1 Score", "Precision", "Recall"
- For stress testing: tell them to multiply numeric features by 1.5 and re-predict

IMPORTANT: Respond with ONLY valid JSON, no markdown:

{{
  "task_0": {{
    "task_description": "Write and execute Python code to load 'data/file.csv'...",
    "expected_output": "Data summary with shape and column types."
  }}
}}
"""


class InstructionWriter:
    """
    Step 4 of the Meta Agent pipeline.
    Generates detailed task descriptions and expected outputs for each agent.
    """

    def __init__(self, llm):
        self.llm = llm

    def write_instructions(
        self,
        subtasks: dict,
        agent_map: dict,
        tool_map: dict,
        dataset_path: str,
        task_description: str,
    ) -> dict:
        """
        Generate detailed instructions for each subtask.
        
        Returns:
            dict mapping subtask_id -> {task_description, expected_output}
        """
        all_subtasks = (
            subtasks.get("modeling_subtasks", [])
            + subtasks.get("mrm_subtasks", [])
        )

        context = []
        for st in all_subtasks:
            sid = st["id"]
            context.append({
                "subtask_id": sid,
                "name": st["name"],
                "description": st["description"],
                "agent_role": agent_map.get(sid, {}).get("role", "Unknown"),
                "tools": tool_map.get(sid, []),
            })

        user_message = (
            f"High-level task: {task_description}\n"
            f"Dataset path: {dataset_path}\n\n"
            f"Subtasks with assigned agents:\n{json.dumps(context, indent=2)}\n\n"
            f"Write detailed instructions for each. Respond with ONLY JSON."
        )

        print("[Meta Agent] Step 4: Writing agent instructions...")

        try:
            response = self._call_llm(INSTRUCTION_PROMPT, user_message)
            instructions = self._parse_json(response)

            print(f"[Meta Agent] Generated instructions for {len(instructions)} tasks")
            return instructions

        except Exception as e:
            print(f"[Meta Agent] Instruction writing failed: {e}")
            print("[Meta Agent] Using fallback instructions...")
            return self._fallback_instructions(
                all_subtasks, agent_map, dataset_path, task_description
            )

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        if hasattr(self.llm, 'invoke'):
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        else:
            full_prompt = f"{system_prompt}\n\n{user_message}"
            response = self.llm.call(full_prompt)
            return str(response)

    def _parse_json(self, response: str) -> dict:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    def _fallback_instructions(
        self, subtasks, agent_map, dataset_path, task_description
    ) -> dict:
        """Generate detailed, dataset-aware instructions from subtask info."""
        instructions = {}

        for st in subtasks:
            sid = st["id"]
            role = agent_map.get(sid, {}).get("role", "Agent")
            name = st["name"]
            desc = st["description"]

            if "data" in name.lower() and ("load" in name.lower() or "extract" in name.lower()):
                task_desc = (
                    f"Write and execute Python code to load '{dataset_path}' using pandas. "
                    f"Replace '?' with NaN using df.replace('?', pd.np.nan if hasattr(pd, 'np') else float('nan')). "
                    f"Print df.shape, df.dtypes, and df.head(). "
                    f"Print the number of missing values per column. "
                    f"Output the shape and column types of the loaded data in the format: "
                    f"'Data shape: (rows, columns), Column types: column1:type, column2:type,...'"
                )
                expected = "Data summary with shape, columns, types, and missing value counts."

            elif "eda" in name.lower() or "exploratory" in name.lower():
                task_desc = (
                    f"Perform exploratory data analysis on '{dataset_path}'. "
                    f"Write Python code using pandas to: "
                    f"1. Load the data and replace '?' with NaN. "
                    f"2. Print shape, dtypes, and missing value counts. "
                    f"3. Print value counts for the target column (last column). "
                    f"4. Print descriptive statistics using df.describe(). "
                    f"5. Identify class balance of the target variable. "
                    f"Print all findings clearly with labels."
                )
                expected = "EDA report with distributions, missing values, class balance, and statistics."

            elif "feature" in name.lower() or "preprocess" in name.lower():
                task_desc = (
                    f"Write and execute Python code to preprocess '{dataset_path}' for binary classification. "
                    f"Steps: "
                    f"1. Load CSV with pandas. "
                    f"2. Replace '?' with NaN, then drop rows with NaN (df.dropna()). "
                    f"3. Use LabelEncoder from sklearn.preprocessing on ALL object/categorical columns. "
                    f"4. Convert all columns to numeric with pd.to_numeric(errors='coerce'). "
                    f"5. Use the LAST column as the target variable y, everything else as X. "
                    f"6. Print the transformed data shape and column types. "
                    f"7. Print 'Preprocessing complete. X shape: ..., y shape: ...'"
                )
                expected = "Transformed dataset summary with shapes and confirmation of preprocessing."

            elif "train" in name.lower() or "model" in name.lower() and "select" in name.lower() or "tuning" in name.lower():
                task_desc = (
                    f"Write and execute Python code to train a classifier on '{dataset_path}'. "
                    f"IMPORTANT - Follow these exact steps: "
                    f"1. import pandas as pd, numpy as np "
                    f"2. from sklearn.model_selection import train_test_split "
                    f"3. from sklearn.ensemble import RandomForestClassifier "
                    f"4. from sklearn.preprocessing import LabelEncoder "
                    f"5. from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score "
                    f"6. Load '{dataset_path}' with pd.read_csv(). "
                    f"7. Replace '?' with NaN: df = df.replace('?', np.nan) "
                    f"8. Drop NaN rows: df = df.dropna() "
                    f"9. Encode categorical columns: for col in df.select_dtypes(include='object').columns: df[col] = LabelEncoder().fit_transform(df[col]) "
                    f"10. Set X = df.iloc[:, :-1], y = df.iloc[:, -1] "
                    f"11. Split: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) "
                    f"12. Train: model = RandomForestClassifier(n_estimators=100, random_state=42); model.fit(X_train, y_train) "
                    f"13. Predict: y_pred = model.predict(X_test) "
                    f"14. Print EXACTLY these lines: "
                    f"    print(f'Accuracy: {{accuracy_score(y_test, y_pred):.4f}}') "
                    f"    print(f'F1 Score: {{f1_score(y_test, y_pred, average=\"weighted\"):.4f}}') "
                    f"    print(f'Precision: {{precision_score(y_test, y_pred, average=\"weighted\"):.4f}}') "
                    f"    print(f'Recall: {{recall_score(y_test, y_pred, average=\"weighted\"):.4f}}') "
                    f"DO NOT generate synthetic/fake data. Use ONLY the real dataset."
                )
                expected = "Model: RandomForest, Metrics: Accuracy=X.XX, F1 Score=X.XX, Precision=X.XX, Recall=X.XX"

            elif "evaluat" in name.lower():
                task_desc = (
                    f"Evaluate the trained model on the test set from '{dataset_path}'. "
                    f"Print Accuracy, F1 Score, Precision, and Recall. "
                    f"Use sklearn.metrics functions. {desc}"
                )
                expected = "Evaluation metrics: Accuracy, F1 Score, Precision, Recall."

            elif "document" in name.lower() or "model card" in name.lower():
                task_desc = (
                    f"Write a Model Card based on the metrics from previous tasks. "
                    f"Include sections: Model Name, Dataset, Task Description, "
                    f"Preprocessing Steps, Algorithm Used, Performance Metrics, "
                    f"Limitations, and Intended Use. {desc}"
                )
                expected = "Text Model Card with all required sections."

            elif "compliance" in name.lower():
                task_desc = (
                    f"Review the modeling output for documentation completeness. "
                    f"Check if these sections exist: data sources, model methodology, "
                    f"performance metrics (Accuracy, F1), and model limitations. "
                    f"Output a compliance report with status for each section. {desc}"
                )
                expected = "Compliance report with pass/fail for each required section."

            elif "stress" in name.lower() or "replicat" in name.lower():
                task_desc = (
                    f"Write and execute Python code to replicate and stress test the model: "
                    f"1. Load '{dataset_path}' with pandas. "
                    f"2. Replace '?' with NaN, drop NaN rows. "
                    f"3. LabelEncoder on object columns. "
                    f"4. X = df.iloc[:, :-1], y = df.iloc[:, -1]. "
                    f"5. Split 80/20 with random_state=42. "
                    f"6. Train RandomForestClassifier(n_estimators=100, random_state=42). "
                    f"7. Print replicated Accuracy and F1 Score. "
                    f"8. STRESS TEST: multiply all X_test numeric values by 1.5. "
                    f"9. Re-predict and print stressed Accuracy. "
                    f"10. Print 'Stress Test Passed' if stressed accuracy > 0.5, else 'Stress Test Warning'. "
                    f"11. Print 'Replication: PASS' if replicated accuracy matches original within 0.01."
                )
                expected = "Replicated metrics, stress test result (Passed/Warning), and replication status."

            elif "soundness" in name.lower() or "validation" in name.lower() or "feature importance" in name.lower():
                task_desc = (
                    f"Assess the conceptual soundness of using a Random Forest classifier "
                    f"for credit card approval prediction. Evaluate: "
                    f"1. Is Random Forest appropriate for this binary classification task? "
                    f"2. Are the features reasonable predictors of credit approval? "
                    f"3. Is the model interpretable enough for financial regulation? "
                    f"4. What are the risks of using this model in production? "
                    f"Output a soundness report with: Conceptual soundness: [sound/unsound], "
                    f"Findings, and Recommendations."
                )
                expected = "Conceptual soundness report with findings and recommendations."

            elif "verdict" in name.lower() or "approval" in name.lower():
                task_desc = (
                    f"Based on the compliance check, stress test, and model validation results, "
                    f"issue a FINAL VERDICT. "
                    f"If accuracy > 0.7 and stress test passed and documentation is compliant, "
                    f"verdict is APPROVED. Otherwise REJECTED. "
                    f"Provide clear reasoning for the decision."
                )
                expected = "Final verdict: APPROVED or REJECTED with detailed reasoning."

            else:
                task_desc = desc
                expected = f"Completed output for: {name}"

            instructions[sid] = {
                "task_description": task_desc,
                "expected_output": expected,
            }

        return instructions