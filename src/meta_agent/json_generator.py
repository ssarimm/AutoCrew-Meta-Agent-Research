import json
import os
from datetime import datetime


class JSONGenerator:
    """
    Step 5 of the Meta Agent pipeline.
    Assembles all outputs from previous steps into a final crew_config.json.
    """

    def generate(
        self,
        task_description: str,
        dataset_path: str,
        llm_provider: str,
        subtasks: dict,
        agent_map: dict,
        tool_map: dict,
        instructions: dict,
    ) -> dict:
        """
        Assemble the final JSON crew configuration.
        
        Returns:
            Complete crew config dict (also saves to file).
        """
        print("[Meta Agent] Step 5: Generating crew config JSON...")

        config = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "AutoCrew Meta Agent",
                "task_description": task_description,
                "dataset": dataset_path,
                "llm_provider": llm_provider,
            },
            "modeling_crew": self._build_crew_config(
                subtasks.get("modeling_subtasks", []),
                agent_map,
                tool_map,
                instructions,
            ),
            "mrm_crew": self._build_crew_config(
                subtasks.get("mrm_subtasks", []),
                agent_map,
                tool_map,
                instructions,
            ),
        }

        # Save to file
        filepath = self._save_config(config)
        print(f"[Meta Agent] Config saved to: {filepath}")

        return config

    def _build_crew_config(
        self, subtask_list, agent_map, tool_map, instructions
    ) -> dict:
        """Build the config for one crew (modeling or mrm)."""
        agents = []
        tasks = []

        for st in subtask_list:
            sid = st["id"]
            agent_info = agent_map.get(sid, {})
            tool_names = tool_map.get(sid, [])
            instr = instructions.get(sid, {})

            # Agent definition
            agent_id = sid.replace("task_", "agent_").replace("mrm_task_", "mrm_agent_")
            agents.append({
                "id": agent_id,
                "role": agent_info.get("role", "Agent"),
                "goal": agent_info.get("goal", st["name"]),
                "backstory": agent_info.get("backstory", "You complete tasks efficiently."),
                "tools": tool_names,
                "allow_delegation": False,
            })

            # Task definition
            context_from = [
                dep.replace("task_", "task_idx_").replace("mrm_task_", "mrm_task_idx_")
                for dep in st.get("depends_on", [])
            ]
            # Map dependency IDs to task indices
            id_to_idx = {s["id"]: i for i, s in enumerate(subtask_list)}
            context_indices = [
                id_to_idx[dep]
                for dep in st.get("depends_on", [])
                if dep in id_to_idx
            ]

            tasks.append({
                "description": instr.get("task_description", st["description"]),
                "expected_output": instr.get("expected_output", "Task output."),
                "agent_id": agent_id,
                "context_from_indices": context_indices,
            })

        return {
            "process": "sequential",
            "agents": agents,
            "tasks": tasks,
        }

    def _save_config(self, config: dict) -> str:
        """Save config to configs/generated/ directory."""
        os.makedirs("configs/generated", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"configs/generated/crew_config_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return filename
