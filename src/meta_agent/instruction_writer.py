import json


INSTRUCTION_PROMPT = """You are an expert at writing precise instructions for AI agents in financial modeling crews.

Given a subtask and its assigned agent role, write the detailed CrewAI task configuration.

For each subtask, produce:
1. task_description: A detailed, specific instruction for the agent. Include the dataset path if relevant. 
   Be explicit about what code to write, what to print, what format to output.
2. expected_output: What the task output should look like (1-2 sentences).

Rules:
- For code-executing agents: tell them exactly what to code, what libraries to use, what to print
- For analysis agents: tell them what to look for and how to format findings
- For writers: tell them what sections to include in the document
- ALWAYS include the dataset path in descriptions where the agent needs to load data
- Be specific about metric names: "Accuracy", "F1 Score", "Precision", "Recall"

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
        """Generate basic instructions from subtask info."""
        instructions = {}

        for st in subtasks:
            sid = st["id"]
            role = agent_map.get(sid, {}).get("role", "Agent")
            name = st["name"]
            desc = st["description"]

            # Build instruction based on role type
            if "data" in name.lower() and ("load" in name.lower() or "extract" in name.lower()):
                task_desc = (
                    f"Write and execute Python code to load '{dataset_path}'. "
                    f"Print df.shape and df.info(). {desc}"
                )
                expected = "Data summary with shape, columns, and types."

            elif "eda" in name.lower() or "exploratory" in name.lower():
                task_desc = (
                    f"Perform exploratory data analysis on '{dataset_path}'. "
                    f"Check for missing values, class imbalance, data types, "
                    f"and correlations. {desc}"
                )
                expected = "EDA report with key findings."

            elif "feature" in name.lower() or "preprocess" in name.lower():
                task_desc = (
                    f"Write and execute Python code to create a preprocessing pipeline "
                    f"for '{dataset_path}'. Handle missing values, encode categoricals, "
                    f"and scale numerics. {desc}"
                )
                expected = "Preprocessing pipeline code and summary."

            elif "train" in name.lower():
                task_desc = (
                    f"Write and execute Python code to load '{dataset_path}' "
                    f"(DO NOT generate fake data). Handle missing values. "
                    f"Train a Random Forest for: {task_description}. "
                    f"Print 'Accuracy' and 'F1 Score'. {desc}"
                )
                expected = "Metrics (Accuracy, F1 Score) printed to stdout."

            elif "evaluat" in name.lower():
                task_desc = (
                    f"Evaluate the trained model on the test set. "
                    f"Print Accuracy, F1 Score, Precision, and Recall. {desc}"
                )
                expected = "Evaluation metrics."

            elif "document" in name.lower() or "model card" in name.lower():
                task_desc = (
                    f"Write a Model Card based on the metrics from previous tasks. "
                    f"Include: model name, dataset, task, metrics, limitations. {desc}"
                )
                expected = "Text Model Card."

            elif "compliance" in name.lower():
                task_desc = f"Review the Model Card for completeness. {desc}"
                expected = "Compliance check result."

            elif "stress" in name.lower():
                task_desc = (
                    f"Write and execute Python code to load the data, "
                    f"multiply numeric columns by 1.5, and check if the model "
                    f"still produces predictions. Print 'Stress Test Passed' or details. {desc}"
                )
                expected = "Stress test logs."

            elif "verdict" in name.lower() or "approval" in name.lower():
                task_desc = (
                    f"Based on the stress test and compliance results, "
                    f"issue a FINAL VERDICT: APPROVED or REJECTED with reasoning. {desc}"
                )
                expected = "Final verdict with reasoning."

            else:
                task_desc = desc
                expected = f"Completed output for: {name}"

            instructions[sid] = {
                "task_description": task_desc,
                "expected_output": expected,
            }

        return instructions
