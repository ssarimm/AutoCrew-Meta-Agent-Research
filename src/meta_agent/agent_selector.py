import json


AGENT_SELECT_PROMPT = """You are an expert at assigning AI agent roles for financial modeling teams.

Given a list of subtasks, assign the best agent role for each subtask.

Available agent roles and their specializations:
- Data Analyst: data loading, data extraction, data splitting, data validation
- Data Scientist: EDA, statistical analysis, data profiling, correlation analysis
- Senior Data Scientist: feature engineering, preprocessing pipelines, encoding, imputation
- ML Engineer: model training, hyperparameter tuning, model selection
- Senior ML Engineer: model evaluation, performance metrics, model saving
- Technical Writer: documentation, model cards, technical reports
- Compliance Officer: documentation review, regulatory checks, procedure verification
- Stress Testing Engineer: robustness testing, adversarial inputs, stress scenarios
- Model Validation Analyst: conceptual soundness, feature importance, interpretability
- Chief Risk Officer: final approval, risk assessment, verdict decisions

For each subtask, assign:
1. role: the agent role from the list above
2. goal: a concise goal statement for the agent
3. backstory: a 1-2 sentence backstory giving the agent context

IMPORTANT: Respond with ONLY valid JSON array, no markdown, no explanation:

[
  {
    "subtask_id": "task_0",
    "role": "Data Analyst",
    "goal": "Load and validate the dataset",
    "backstory": "You ensure data quality before any modeling begins."
  }
]
"""


class AgentSelector:
    """
    Step 2 of the Meta Agent pipeline.
    Takes decomposed subtasks and assigns an agent role to each.
    """

    def __init__(self, llm):
        self.llm = llm

    def select_agents(self, subtasks: dict) -> dict:
        """
        Assign agent roles to each subtask.
        
        Args:
            subtasks: dict with "modeling_subtasks" and "mrm_subtasks"
            
        Returns:
            dict mapping subtask_id -> agent config
        """
        all_subtasks = (
            subtasks.get("modeling_subtasks", [])
            + subtasks.get("mrm_subtasks", [])
        )

        subtask_text = json.dumps(all_subtasks, indent=2)
        user_message = (
            f"Here are the subtasks:\n{subtask_text}\n\n"
            f"Assign agent roles. Respond with ONLY JSON."
        )

        print("[Meta Agent] Step 2: Selecting agents for subtasks...")

        try:
            response = self._call_llm(AGENT_SELECT_PROMPT, user_message)
            assignments = self._parse_json_array(response)

            # Build lookup dict
            agent_map = {}
            for assignment in assignments:
                sid = assignment["subtask_id"]
                agent_map[sid] = {
                    "role": assignment["role"],
                    "goal": assignment["goal"],
                    "backstory": assignment["backstory"],
                }

            print(f"[Meta Agent] Assigned {len(agent_map)} agents")
            return agent_map

        except Exception as e:
            print(f"[Meta Agent] Agent selection failed: {e}")
            print("[Meta Agent] Using fallback agent assignments...")
            return self._fallback_assignments(all_subtasks)

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call the LLM and return raw text response."""
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

    def _parse_json_array(self, response: str) -> list:
        """Parse JSON array from LLM response."""
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    def _fallback_assignments(self, subtasks: list) -> dict:
        """Fallback role assignments based on subtask name keywords."""
        role_map = {
            "data loading": ("Data Analyst", "Load and validate data", "You ensure data quality."),
            "data extraction": ("Data Analyst", "Extract and split data", "You handle data loading."),
            "eda": ("Data Scientist", "Perform exploratory analysis", "You uncover data patterns."),
            "exploratory": ("Data Scientist", "Analyze data characteristics", "You profile datasets."),
            "feature": ("Senior Data Scientist", "Create preprocessing pipeline", "You engineer features."),
            "preprocess": ("Senior Data Scientist", "Build data pipeline", "You prepare data for modeling."),
            "model train": ("ML Engineer", "Train the model", "You build ML models."),
            "training": ("ML Engineer", "Train and save model", "You write training code."),
            "tuning": ("ML Engineer", "Tune hyperparameters", "You optimize model performance."),
            "selection": ("ML Engineer", "Select best model", "You compare model architectures."),
            "evaluat": ("Senior ML Engineer", "Evaluate model performance", "You compute metrics."),
            "document": ("Technical Writer", "Write documentation", "You create model cards."),
            "model card": ("Technical Writer", "Summarize findings", "You document technical results."),
            "compliance": ("Compliance Officer", "Check documentation completeness", "You verify procedures."),
            "stress": ("Stress Testing Engineer", "Test model robustness", "You simulate extreme conditions."),
            "soundness": ("Model Validation Analyst", "Assess model validity", "You check interpretability."),
            "replicat": ("ML Engineer", "Replicate model training", "You verify reproducibility."),
            "verdict": ("Chief Risk Officer", "Issue final approval", "You make the final call."),
            "approval": ("Chief Risk Officer", "Approve or reject", "You review all evidence."),
        }

        agent_map = {}
        for subtask in subtasks:
            sid = subtask["id"]
            name_lower = subtask["name"].lower()
            desc_lower = subtask.get("description", "").lower()
            combined = name_lower + " " + desc_lower

            matched = False
            for keyword, (role, goal, backstory) in role_map.items():
                if keyword in combined:
                    agent_map[sid] = {
                        "role": role,
                        "goal": goal,
                        "backstory": backstory,
                    }
                    matched = True
                    break

            if not matched:
                agent_map[sid] = {
                    "role": "ML Engineer",
                    "goal": subtask["name"],
                    "backstory": "You handle general ML tasks.",
                }

        return agent_map
