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
- CRITICAL: Each agent must complete its entire job in ONE single code execution call, not multiple.
- CRITICAL: For data loading from 'data/credit_card_approval.csv', ALWAYS use pd.read_csv(..., header=None) because this CSV has NO header row.
- CRITICAL: For data loading from 'data/cs-training.csv', use df.fillna(df.median(numeric_only=True)) instead of dropna().
- CRITICAL: For data loading from 'data/creditcard_2023.csv', use class_weight='balanced' in RandomForestClassifier. This dataset has 568K rows — use n_estimators=20, max_depth=15, n_jobs=-1 to avoid timeout. Downsample training majority class to 50K rows for speed.
- CRITICAL: For 'data/credit_card_approval.csv', replace '?' with NaN, drop NaN rows, use LabelEncoder on object columns.
- CRITICAL: NEVER fabricate, hallucinate, or invent metric values. Only report numbers that were actually printed by executed code. If code times out or fails, report the error — do NOT make up results.
- CRITICAL: ALWAYS use average='weighted' for f1_score, precision_score, and recall_score. NEVER use the default binary average.
- CRITICAL: NEVER use f-strings in print statements. Use: print('Accuracy:', round(accuracy, 4))
- CRITICAL: Use ONLY single quotes for string literals in Python code. Double quotes break JSON serialization.
- CRITICAL: Always handle NaN on the FULL DataFrame BEFORE splitting into X and y.
- CRITICAL: Do NOT use StandardScaler. RandomForestClassifier does not require feature scaling.
- CRITICAL: Tell agents to NEVER generate synthetic or fake data.
- For documentation/model card tasks: must state Algorithm name, Dataset path, and all Performance Metrics.
- For verdict/approval tasks: base APPROVED/REJECTED on model performance (Accuracy > 0.7 and Stress Test results).
- For stress testing: multiply numeric features by 1.5, re-predict, and print "Stress Test Passed" or "Stress Test Warning"

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
                max_tokens=2500,
            )
            return response.choices[0].message.content

    def _parse_json(self, response: str) -> dict:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    def _get_load_code(self, dataset_path: str) -> str:
        if "credit_card_approval" in dataset_path:
            return (
                f"df = pd.read_csv('{dataset_path}', header=None)\n"
                f"df = df.replace('?', np.nan)\n"
                f"df = df.dropna()\n"
                f"for col in df.select_dtypes(include='object').columns:\n"
                f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
            )
        elif "cs-training" in dataset_path:
            return (
                f"df = pd.read_csv('{dataset_path}')\n"
                f"df = df.drop(columns=[c for c in df.columns if 'unnamed' in c.lower()], errors='ignore')\n"
                f"df = df.fillna(df.median(numeric_only=True))\n"
                f"for col in df.select_dtypes(include='object').columns:\n"
                f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
            )
        elif "creditcard_2023" in dataset_path:
            return (
                f"df = pd.read_csv('{dataset_path}')\n"
                f"df = df.drop(columns=[c for c in df.columns if c.lower() in ('id', 'unnamed: 0')], errors='ignore')\n"
                f"df = df.dropna()\n"
                f"for col in df.select_dtypes(include='object').columns:\n"
                f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
            )
        else:
            return (
                f"df = pd.read_csv('{dataset_path}')\n"
                f"df = df.replace('?', np.nan)\n"
                f"df = df.dropna()\n"
                f"for col in df.select_dtypes(include='object').columns:\n"
                f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
            )

    def _get_target_code(self, dataset_path: str) -> str:
        if "credit_card_approval" in dataset_path:
            return (
                f"target_col = df.columns[-1]\n"
                f"X = df.drop(columns=[target_col])\n"
                f"y = df[target_col]\n"
            )
        elif "cs-training" in dataset_path:
            return (
                f"target_col = 'SeriousDlqin2yrs'\n"
                f"X = df.drop(columns=[target_col])\n"
                f"y = df[target_col]\n"
            )
        elif "creditcard_2023" in dataset_path:
            return (
                f"target_col = 'Class'\n"
                f"X = df.drop(columns=[target_col])\n"
                f"y = df[target_col]\n"
            )
        else:
            return (
                f"known = {{'SeriousDlqin2yrs', 'Class'}}\n"
                f"target_col = next((c for c in known if c in df.columns), df.columns[-1])\n"
                f"X = df.drop(columns=[target_col] + [c for c in df.columns if 'unnamed' in c.lower()])\n"
                f"y = df[target_col]\n"
            )

    def _get_rf_params(self, dataset_path: str) -> str:
        if "creditcard_2023" in dataset_path:
            return "n_estimators=20, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced'"
        return "n_estimators=100, random_state=42"

    def _get_sampling_code(self, dataset_path: str) -> str:
        """Return training set downsampling code for large datasets."""
        if "creditcard_2023" in dataset_path:
            return (
                f"# Downsample majority class to avoid timeout on large datasets\n"
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
        return ""

    def _fallback_instructions(
        self, subtasks, agent_map, dataset_path, task_description=None
    ) -> dict:
        instructions = {}

        load_code = self._get_load_code(dataset_path)
        target_code = self._get_target_code(dataset_path)
        rf_params = self._get_rf_params(dataset_path)
        sampling_code = self._get_sampling_code(dataset_path)

        header_param = ", header=None" if "credit_card_approval" in dataset_path else ""

        for st in subtasks:
            sid = st["id"]
            name = st["name"]
            desc = st["description"]
            combined = (name + " " + desc).lower()

            # MRM-specific checks first
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
                    f"NEVER fabricate results — only report numbers from actual code execution. "
                    f"Steps (all in ONE script): "
                    f"1. import pandas as pd, numpy as np, from sklearn.model_selection import train_test_split, "
                    f"from sklearn.ensemble import RandomForestClassifier, from sklearn.preprocessing import LabelEncoder, "
                    f"from sklearn.metrics import accuracy_score, f1_score "
                    f"2. Load and preprocess:\n{load_code}"
                    f"3. Target detection:\n{target_code}"
                    f"4. X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) "
                    f"{sampling_code}"
                    f"5. model = RandomForestClassifier({rf_params}); model.fit(X_train, y_train) "
                    f"6. y_pred = model.predict(X_test); print('Baseline Accuracy:', round(accuracy_score(y_test, y_pred), 4)) "
                    f"7. X_test_s = X_test.copy() "
                    f"   for col in X_test_s.select_dtypes(include=[np.number]).columns: X_test_s[col] = X_test_s[col] * 1.5 "
                    f"   y_pred_s = model.predict(X_test_s) "
                    f"8. stressed_acc = accuracy_score(y_test, y_pred_s); print('Stressed Accuracy:', round(stressed_acc, 4)) "
                    f"9. print('Stress Test Passed') if stressed_acc > 0.5 else print('Stress Test Warning')"
                )
                expected = "Baseline Accuracy, Stressed Accuracy, Stress Test Passed or Warning."

            elif "soundness" in combined or ("validation" in combined and "model" in combined) or "feature importance" in combined:
                task_desc = (
                    f"Assess the conceptual soundness of the ML model used for: "
                    f"'{task_description or desc}'. Dataset: '{dataset_path}'. "
                    f"Output: Conceptual soundness: [sound/unsound], Findings, Recommendations."
                )
                expected = "Conceptual soundness report with findings and recommendations."

            elif "verdict" in combined or ("approval" in combined and "final" in combined):
                task_desc = (
                    f"Issue a FINAL VERDICT based on the stress test results. "
                    f"VERDICT RULE: Base APPROVED/REJECTED on model performance ONLY — not on documentation gaps. "
                    f"- If Baseline Accuracy > 0.7 AND 'Stress Test Passed': FINAL VERDICT: APPROVED "
                    f"- If stress test failed to run: FINAL VERDICT: INCONCLUSIVE "
                    f"- Otherwise: FINAL VERDICT: REJECTED "
                    f"State your verdict clearly and give 1-2 sentences of reasoning."
                )
                expected = "FINAL VERDICT: APPROVED or REJECTED or INCONCLUSIVE with reasoning."

            elif "data" in combined and ("load" in combined or "extract" in combined):
                task_desc = (
                    f"Write and execute Python code to load '{dataset_path}' using pandas{' with header=None' if header_param else ''}. "
                    f"Print df.shape, list(df.columns), df.dtypes, and df.isnull().sum()."
                )
                expected = "Data summary with shape, columns, types, and missing value counts."

            elif "eda" in combined or "exploratory" in combined:
                task_desc = (
                    f"Perform exploratory data analysis on '{dataset_path}' using ONLY pandas "
                    f"(NO matplotlib, NO seaborn — text output only). "
                    f"1. df = pd.read_csv('{dataset_path}'{header_param}) "
                    f"2. print(df.shape); print(df.dtypes); print(df.isnull().sum()) "
                    f"3. print(df.describe()); print(df.iloc[:, -1].value_counts())"
                )
                expected = "EDA report with shape, dtypes, missing values, descriptive stats, class balance."

            elif "feature" in combined or "preprocess" in combined:
                task_desc = (
                    f"Write and execute Python code to preprocess '{dataset_path}' for binary classification. "
                    f"1. import pandas as pd, numpy as np; from sklearn.preprocessing import LabelEncoder "
                    f"2. Load and preprocess:\n{load_code}"
                    f"3. Target detection:\n{target_code}"
                    f"4. print('Preprocessing complete. X shape:', X.shape, 'y shape:', y.shape)"
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
                    f"2. Load and preprocess:\n{load_code}"
                    f"3. Target detection:\n{target_code}"
                    f"4. X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) "
                    f"{sampling_code}"
                    f"5. model = RandomForestClassifier({rf_params}); model.fit(X_train, y_train) "
                    f"6. y_pred = model.predict(X_test) "
                    f"7. print('Accuracy:', round(accuracy_score(y_test, y_pred), 4)) "
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
