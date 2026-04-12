import json


INSTRUCTION_PROMPT = """You are an expert at writing precise instructions for AI agents in financial modeling crews.

Given a subtask and its assigned agent role, write the detailed CrewAI task configuration.

For each subtask, produce:
1. task_description: A detailed, specific instruction for the agent. Include the dataset path if relevant. 
   Be explicit about what code to write, what to print, what format to output.
2. expected_output: What the task output should look like (1-2 sentences).

Rules:
- For code-executing agents: tell them EXACTLY what to code step by step, what libraries to import, what to print
- CRITICAL: Every code execution is a completely fresh process. Each script MUST start by importing all libraries and loading the dataset from its path. Never reference variables from a previous tool call — they do not exist.
- CRITICAL: Each agent must complete its entire job in ONE single code execution call, not multiple. Load data, process, train, and print results all in one script.
- CRITICAL: For data loading, ALWAYS include: replace '?' with NaN, drop NaN rows, use LabelEncoder on object columns
- CRITICAL: Before using ANY sklearn transformer (StandardScaler, PolynomialFeatures, etc.), ALWAYS handle NaN first: df.dropna() or df.fillna(df.median(numeric_only=True)). Never pass NaN values to sklearn.
- CRITICAL: For model training, choose an appropriate sklearn classification algorithm based on the task (e.g. RandomForestClassifier, GradientBoostingClassifier, LogisticRegression). Print Accuracy, F1 Score, Precision, and Recall using sklearn.metrics.
- CRITICAL: For the target column — if the task description explicitly names it (e.g. 'SeriousDlqin2yrs', 'Class'), use that exact name. If the task description does NOT name the target column, detect it dynamically with: known = {'SeriousDlqin2yrs', 'Class'}; target_col = next((c for c in known if c in df.columns), df.columns[-1]). NEVER invent column names like 'credit_approved', 'label', 'target', 'outcome', 'approved', 'p', 'q' — these will cause a KeyError and crash the script. Also drop any index columns (e.g. 'Unnamed: 0') before splitting X and y.
- CRITICAL: NEVER trust column names from the context or outputs of previous tasks — prior agents may hallucinate renamed columns (e.g. calling the last column 'p' instead of its real name '+'). Always determine column names by loading the CSV file directly in your script and reading df.columns at runtime.
- CRITICAL: For ANY task involving model evaluation, metrics reporting, or model card creation: the script MUST include the FULL pipeline — load raw CSV, replace '?', dropna, encode categoricals, split X/y, train a fresh RandomForestClassifier, predict, and print metrics. NEVER write code that assumes a saved model or preprocessed data from a previous task exists. Every script starts with a blank slate.
- CRITICAL: Tell agents to NEVER generate synthetic or fake data — not even as a fallback if the file is not found. Always use the exact dataset path provided. If the file is missing, the agent must report the error and stop.
- CRITICAL: Always embed the exact dataset path (e.g. 'data/cs-training.csv') directly in the task_description string. Never write vague instructions like "load the data".
- CRITICAL: For EDA/analysis tasks: if the agent has the exploratory_data_analysis tool, tell it to use ONLY that tool and stop — do NOT also call execute_python_code for EDA. Using both wastes tokens and causes rate limit errors. If the agent only has execute_python_code, use ONLY pandas text output (print statements), no matplotlib/seaborn.
- CRITICAL: NEVER use f-strings (f"...{var}...") in ANY print statement inside task_description. F-string curly braces break JSON serialization. Instead use: print("Accuracy:", round(accuracy, 4)) or print("Accuracy: " + str(round(accuracy, 4))). This applies to ALL metric prints.
- CRITICAL: In ALL Python code inside task_description, use ONLY single quotes for string literals. NEVER use double quotes. Write pd.read_csv('data/file.csv') NOT pd.read_csv("data/file.csv"). Double quotes inside a JSON string value break the tool input parser, causing SyntaxError. This applies to every string in every line of code.
- CRITICAL: Always call df.dropna() on the FULL DataFrame BEFORE splitting into X and y. NEVER call X.dropna() or y.dropna() separately — this misaligns row counts and causes ValueError in train_test_split.
- CRITICAL: Do NOT use StandardScaler or any feature scaler. RandomForestClassifier does not require feature scaling, and adding a scaler after separate dropna causes row count mismatches.
- For documentation/model card tasks: the model card MUST explicitly state the Algorithm name used, Dataset path, and all Performance Metrics (Accuracy, F1 Score, Precision, Recall). These fields are required for MRM compliance checks.
- For verdict/approval tasks: base the APPROVED/REJECTED decision primarily on model performance (Accuracy > 0.7 and Stress Test results). Documentation compliance is informational — do NOT reject a well-performing model solely due to minor documentation gaps.
- For analysis agents: tell them what to look for and how to format findings
- ALWAYS include the dataset path in descriptions where the agent needs to load data
- Be specific about metric names: "Accuracy", "F1 Score", "Precision", "Recall"
- For stress testing: tell them to multiply numeric features by 1.5, re-predict, and print "Stress Test Passed" or "Stress Test Warning"

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
            import litellm
            response = litellm.completion(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=getattr(self.llm, 'temperature', 0.2),
                api_key=getattr(self.llm, 'api_key', None),
                base_url=getattr(self.llm, 'base_url', None),
                max_tokens=2500,  # instructions: 7 tasks × ~300 tokens each — extra headroom to avoid mid-JSON truncation
            )
            return response.choices[0].message.content

    def _parse_json(self, response: str) -> dict:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    def _fallback_instructions(
        self, subtasks, agent_map, dataset_path, task_description=None
    ) -> dict:
        """Generate detailed, dataset-aware instructions from subtask info."""
        instructions = {}

        for st in subtasks:
            sid = st["id"]
            name = st["name"]
            desc = st["description"]
            combined = (name + " " + desc).lower()

            # MRM-specific checks come FIRST to avoid mismatching with generic keywords
            if "compliance" in combined or ("review" in combined and "document" in combined):
                task_desc = (
                    f"Review the modeling output for MRM documentation completeness. "
                    f"Check if these fields are present: algorithm name, dataset path, "
                    f"performance metrics (Accuracy, F1 Score), and model limitations. "
                    f"Output: PASS if all present, FAIL with list of missing items."
                )
                expected = "Compliance check: PASS or FAIL with list of missing items."

            elif "stress" in combined or "replicat" in combined:
                task_desc = (
                    f"Write and execute a complete Python script to stress test the model on '{dataset_path}'. "
                    f"Steps (all in ONE script): "
                    f"1. import pandas as pd, numpy as np, from sklearn.model_selection import train_test_split, "
                    f"from sklearn.ensemble import RandomForestClassifier, from sklearn.preprocessing import LabelEncoder, "
                    f"from sklearn.metrics import accuracy_score, f1_score "
                    f"2. df = pd.read_csv('{dataset_path}'); df = df.replace('?', np.nan); df = df.dropna() "
                    f"3. Encode all object columns with LabelEncoder "
                    f"4. Detect target column: known = {{'SeriousDlqin2yrs', 'Class'}}; "
                    f"target_col = next((c for c in known if c in df.columns), df.columns[-1]); "
                    f"X = df.drop(columns=[target_col] + [c for c in df.columns if 'unnamed' in c.lower()]); y = df[target_col] "
                    f"5. X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) "
                    f"6. model = RandomForestClassifier(n_estimators=100, random_state=42); model.fit(X_train, y_train) "
                    f"7. y_pred = model.predict(X_test); print('Baseline Accuracy:', accuracy_score(y_test, y_pred)) "
                    f"8. X_test_s = X_test.copy(); X_test_s *= 1.5; y_pred_s = model.predict(X_test_s) "
                    f"9. stressed_acc = accuracy_score(y_test, y_pred_s); print('Stressed Accuracy:', stressed_acc) "
                    f"10. print('Stress Test Passed') if stressed_acc > 0.5 else print('Stress Test Warning')"
                )
                expected = "Baseline Accuracy, Stressed Accuracy, Stress Test Passed or Warning."

            elif "soundness" in combined or ("validation" in combined and "model" in combined) or "feature importance" in combined:
                task_desc = (
                    f"Assess the conceptual soundness of the ML model used for the following task: "
                    f"'{task_description or desc}'. Dataset: '{dataset_path}'. Evaluate: "
                    f"1. Is the chosen algorithm appropriate for this binary classification task? "
                    f"2. Are the dataset features reasonable predictors for the target variable? "
                    f"3. Is the model interpretable enough for financial regulation compliance? "
                    f"4. What are the production risks of deploying this model? "
                    f"Output: Conceptual soundness: [sound/unsound], Findings, Recommendations."
                )
                expected = "Conceptual soundness report with findings and recommendations."

            elif "verdict" in combined or ("approval" in combined and "final" in combined):
                task_desc = (
                    f"Issue a FINAL VERDICT based on the stress test results. "
                    f"VERDICT RULE: Base APPROVED/REJECTED on model performance ONLY — not on documentation gaps. "
                    f"- If Baseline Accuracy > 0.7 AND 'Stress Test Passed': FINAL VERDICT: APPROVED "
                    f"- Otherwise: FINAL VERDICT: REJECTED "
                    f"State your verdict clearly and give 1-2 sentences of reasoning."
                )
                expected = "FINAL VERDICT: APPROVED or REJECTED with reasoning."

            # Modeling-specific checks
            elif "data" in combined and ("load" in combined or "extract" in combined):
                task_desc = (
                    f"Write and execute Python code to load '{dataset_path}' using pandas. "
                    f"Replace '?' with NaN: df = df.replace('?', float('nan')). "
                    f"Print df.shape, list(df.columns), df.dtypes, and df.isnull().sum()."
                )
                expected = "Data summary with shape, columns, types, and missing value counts."

            elif "eda" in combined or "exploratory" in combined:
                task_desc = (
                    f"Perform exploratory data analysis on '{dataset_path}' using ONLY pandas "
                    f"(NO matplotlib, NO seaborn — text output only). "
                    f"1. df = pd.read_csv('{dataset_path}'); df = df.replace('?', float('nan')) "
                    f"2. print(df.shape); print(df.dtypes); print(df.isnull().sum()) "
                    f"3. print(df.describe()); print(df.iloc[:, -1].value_counts())"
                )
                expected = "EDA report with shape, dtypes, missing values, descriptive stats, class balance."

            elif "feature" in combined or "preprocess" in combined:
                task_desc = (
                    f"Write and execute Python code to preprocess '{dataset_path}' for binary classification. "
                    f"1. import pandas as pd, numpy as np; from sklearn.preprocessing import LabelEncoder "
                    f"2. df = pd.read_csv('{dataset_path}'); df = df.replace('?', np.nan); df = df.dropna() "
                    f"3. LabelEncoder on all object columns "
                    f"4. Detect target: known = {{'SeriousDlqin2yrs', 'Class'}}; "
                    f"target_col = next((c for c in known if c in df.columns), df.columns[-1]); "
                    f"X = df.drop(columns=[target_col] + [c for c in df.columns if 'unnamed' in c.lower()]); y = df[target_col] "
                    f"5. print('Preprocessing complete. X shape:', X.shape, 'y shape:', y.shape)"
                )
                expected = "Preprocessing confirmation with X and y shapes."

            elif "train" in combined or ("model" in combined and "select" in combined) or "tuning" in combined or "evaluat" in combined:
                task_desc = (
                    f"Write and execute a complete self-contained Python script on '{dataset_path}'. "
                    f"1. import pandas as pd, numpy as np "
                    f"   from sklearn.model_selection import train_test_split "
                    f"   from sklearn.ensemble import RandomForestClassifier "
                    f"   from sklearn.preprocessing import LabelEncoder "
                    f"   from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score "
                    f"2. df = pd.read_csv('{dataset_path}'); df = df.replace('?', np.nan); df = df.dropna() "
                    f"3. Encode all object columns with LabelEncoder "
                    f"4. known = {{'SeriousDlqin2yrs', 'Class'}}; "
                    f"target_col = next((c for c in known if c in df.columns), df.columns[-1]); "
                    f"X = df.drop(columns=[target_col] + [c for c in df.columns if 'unnamed' in c.lower()]); y = df[target_col] "
                    f"5. X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) "
                    f"6. model = RandomForestClassifier(n_estimators=100, random_state=42); model.fit(X_train, y_train) "
                    f"7. y_pred = model.predict(X_test) "
                    f"8. print('Accuracy:', round(accuracy_score(y_test, y_pred), 4)) "
                    f"   print('F1 Score:', round(f1_score(y_test, y_pred, average='weighted'), 4)) "
                    f"   print('Precision:', round(precision_score(y_test, y_pred, average='weighted'), 4)) "
                    f"   print('Recall:', round(recall_score(y_test, y_pred, average='weighted'), 4))"
                )
                expected = "Accuracy, F1 Score, Precision, Recall printed to stdout."

            elif "document" in combined or "model card" in combined:
                task_desc = (
                    f"Write a Model Card based on the metrics from previous tasks. "
                    f"REQUIRED fields: Model Name, Algorithm Used, Dataset ({dataset_path}), "
                    f"Task ({task_description or desc}), Accuracy, F1 Score, Precision, Recall, "
                    f"Preprocessing Steps, Limitations, Intended Use. "
                    f"State each field on a separate line."
                )
                expected = "Model Card with all required fields including Algorithm, Dataset, and metrics."

            else:
                task_desc = desc
                expected = f"Completed output for: {name}"

            instructions[sid] = {
                "task_description": task_desc,
                "expected_output": expected,
            }

        return instructions