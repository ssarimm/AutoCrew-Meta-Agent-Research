import json
import os


DECOMPOSE_PROMPT = """You are a task decomposition expert for financial modeling workflows.

Given a high-level task description and a dataset path, break it down into ordered subtasks 
that form a complete ML pipeline.

Consider these standard financial modeling steps:
1. Data Extraction / Loading
2. Exploratory Data Analysis (EDA)
3. Feature Engineering / Preprocessing
4. Model Selection / Hyperparameter Tuning
5. Model Training
6. Model Evaluation
7. Documentation / Model Card Writing

CRITICAL CONSTRAINTS:
- You MUST include a model training step in your modeling_subtasks. A pipeline without training is invalid.
- Generate AT MOST 4 modeling subtasks and 3 MRM subtasks (API rate limit).
- If 4 subtasks are not enough to include all steps, prioritize: Data Loading, Preprocessing, Model Training, Documentation. EDA and Model Selection are optional if slots are limited.
- MRM subtasks MUST include a stress testing subtask (multiply numeric features by 1.5, re-predict, compare accuracy). The stress test must execute real code on the real dataset — it is NOT optional.
- MRM subtasks MUST end with a final verdict subtask that cites stress test accuracy and baseline accuracy.

Also consider these MRM (Model Risk Management) steps:
1. Documentation Compliance Check
2. Model Replication
3. Conceptual Soundness / Feature Importance
4. Stress Testing / Outcome Analysis
5. Final Verdict / Approval

Based on the task, select ONLY the relevant subtasks. Not every task needs all steps.

IMPORTANT: Respond with ONLY valid JSON, no markdown, no explanation. Use this exact format:

{
  "modeling_subtasks": [
    {
      "id": "task_0",
      "name": "short name",
      "description": "what this subtask should accomplish",
      "depends_on": []
    },
    {
      "id": "task_1",
      "name": "short name",
      "description": "what this subtask should accomplish",
      "depends_on": ["task_0"]
    }
  ],
  "mrm_subtasks": [
    {
      "id": "mrm_task_0",
      "name": "short name",
      "description": "what this MRM subtask should accomplish",
      "depends_on": []
    }
  ]
}
"""


class TaskDecomposer:
    """
    Step 1 of the Meta Agent pipeline.
    Takes a natural language task description and breaks it into
    ordered subtasks using an LLM call.
    """

    def __init__(self, llm):
        self.llm = llm

    def decompose(self, task_description: str, dataset_path: str) -> dict:
        """
        Decompose a high-level task into modeling + MRM subtasks.
        
        Args:
            task_description: e.g. "Predict credit approval (Binary Classification)"
            dataset_path: e.g. "data/credit_card_approval.csv"
            
        Returns:
            dict with "modeling_subtasks" and "mrm_subtasks" lists
        """
        user_message = (
            f"Task: {task_description}\n"
            f"Dataset: {dataset_path}\n\n"
            f"Decompose this into subtasks. Respond with ONLY JSON."
        )

        print("\n[Meta Agent] Step 1: Decomposing task...")

        try:
            # Use LLM to decompose
            response = self._call_llm(DECOMPOSE_PROMPT, user_message)
            subtasks = self._parse_json(response)

            n_modeling = len(subtasks.get("modeling_subtasks", []))
            n_mrm = len(subtasks.get("mrm_subtasks", []))
            print(f"[Meta Agent] Decomposed into {n_modeling} modeling + {n_mrm} MRM subtasks")

            return subtasks

        except Exception as e:
            print(f"[Meta Agent] Decomposition failed: {e}")
            print("[Meta Agent] Using fallback decomposition...")
            return self._fallback_decomposition(task_description, dataset_path)

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call the LLM and return raw text response."""
        if hasattr(self.llm, 'invoke'):
            # LangChain-style (Gemini, FakeListChatModel)
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, 'content') else str(response)
        else:
            # Use litellm directly to bypass CrewAI's event bus (prevents display spam)
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
                max_tokens=1200,  # decomposition JSON can be verbose; 600 caused truncation → parse failures
            )
            return response.choices[0].message.content

    def _parse_json(self, response: str) -> dict:
        """Parse JSON from LLM response, handling markdown fences."""
        text = response.strip()

        # Remove markdown code fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        return json.loads(text)

    def _fallback_decomposition(self, task_description: str, dataset_path: str) -> dict:
        """Fallback if LLM decomposition fails. Optimized for rate-limited APIs."""
        return {
            "modeling_subtasks": [
                {
                    "id": "task_0",
                    "name": "Data Loading and EDA",
                    "description": f"Load '{dataset_path}' and perform exploratory data analysis.",
                    "depends_on": [],
                },
                {
                    "id": "task_1",
                    "name": "Model Training",
                    "description": (
                        f"Load '{dataset_path}'. Replace '?' with NaN, drop NaN rows. "
                        f"Use LabelEncoder on all object columns. Use last column as target. "
                        f"Split 80/20. Train a RandomForestClassifier. "
                        f"Print EXACTLY: Accuracy: X.XXXX, F1 Score: X.XXXX, Precision: X.XXXX, Recall: X.XXXX. Task: {task_description}"
                    ),
                    "depends_on": ["task_0"],
                },
                {
                    "id": "task_2",
                    "name": "Documentation",
                    "description": "Write a Model Card summarizing the data, model, and metrics.",
                    "depends_on": ["task_1"],
                },
            ],
            "mrm_subtasks": [
                {
                    "id": "mrm_task_0",
                    "name": "Compliance and Stress Test",
                    "description": "Review the Model Card for completeness. Test model with perturbed inputs.",
                    "depends_on": [],
                },
                {
                    "id": "mrm_task_1",
                    "name": "Final Verdict",
                    "description": "Based on all results, APPROVE or REJECT the model.",
                    "depends_on": ["mrm_task_0"],
                },
            ],
        }
