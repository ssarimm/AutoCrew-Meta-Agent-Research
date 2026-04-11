import json
from src.tools.tool_registry import ToolRegistry


TOOL_ASSIGN_PROMPT = """You are an expert at assigning tools to AI agents for financial modeling.

Given a list of agents with their roles and tasks, assign the appropriate tools to each.

Available tools:
{tool_catalog}

Rules:
- Agents that need to write and execute Python code should get "code_execution"
- Agents doing EDA can get "eda_tool" OR "code_execution" (or both)
- Agents checking compliance documentation should get "cag_tool"
- Agents that only analyze text (like writers, judges) may need NO tools (empty list)
- Do NOT assign tools that are not in the available list

IMPORTANT: Respond with ONLY valid JSON, no markdown, no explanation:

{{
  "task_0": ["code_execution"],
  "task_1": ["eda_tool"],
  "task_2": ["code_execution"],
  "task_3": []
}}
"""


class ToolAssigner:
    """
    Step 3 of the Meta Agent pipeline.
    Takes agent assignments and assigns appropriate tools to each.
    """

    def __init__(self, llm):
        self.llm = llm

    def assign_tools(self, subtasks: dict, agent_map: dict) -> dict:
        """
        Assign tools to each agent/subtask.
        
        Args:
            subtasks: dict with modeling_subtasks and mrm_subtasks
            agent_map: dict mapping subtask_id -> agent config
            
        Returns:
            dict mapping subtask_id -> list of tool name strings
        """
        # Build context for LLM
        all_subtasks = (
            subtasks.get("modeling_subtasks", [])
            + subtasks.get("mrm_subtasks", [])
        )

        agents_info = []
        for st in all_subtasks:
            sid = st["id"]
            agent = agent_map.get(sid, {})
            agents_info.append({
                "subtask_id": sid,
                "subtask_name": st["name"],
                "subtask_description": st["description"],
                "agent_role": agent.get("role", "Unknown"),
            })

        # Get tool catalog
        catalog = ToolRegistry.get_catalog()
        catalog_text = json.dumps(catalog, indent=2)

        prompt = TOOL_ASSIGN_PROMPT.replace("{tool_catalog}", catalog_text)
        user_message = (
            f"Agents and their tasks:\n{json.dumps(agents_info, indent=2)}\n\n"
            f"Assign tools. Respond with ONLY JSON."
        )

        print("[Meta Agent] Step 3: Assigning tools to agents...")

        try:
            response = self._call_llm(prompt, user_message)
            tool_map = self._parse_json(response)

            # Validate: ensure code-writing agents have code_execution
            tool_map = self._validate_tool_assignments(tool_map, agent_map)
            total_tools = sum(len(v) for v in tool_map.values())
            print(f"[Meta Agent] Assigned {total_tools} tools across {len(tool_map)} agents")
            return tool_map

        except Exception as e:
            print(f"[Meta Agent] Tool assignment failed: {e}")
            print("[Meta Agent] Using fallback tool assignments...")
            return self._fallback_tools(agent_map)

    def _validate_tool_assignments(self, tool_map: dict, agent_map: dict) -> dict:
        """Ensure agents that need code execution actually have it."""
        roles_needing_code = {
            "ML Engineer", "Senior ML Engineer", "Data Analyst",
            "Senior Data Scientist", "Stress Testing Engineer"
        }
        for sid, agent in agent_map.items():
            role = agent.get("role", "")
            tools = tool_map.get(sid, [])
            if role in roles_needing_code and "code_execution" not in tools:
                tools.append("code_execution")
                tool_map[sid] = tools
                print(f"[Meta Agent] Auto-added code_execution to {role} ({sid})")
        return tool_map

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

    def _fallback_tools(self, agent_map: dict) -> dict:
        """Fallback: assign tools based on agent role."""
        role_tools = {
            "Data Analyst": ["code_execution"],
            "Data Scientist": ["eda_tool"],
            "Senior Data Scientist": ["code_execution"],
            "ML Engineer": ["code_execution"],
            "Senior ML Engineer": ["code_execution"],
            "Technical Writer": [],
            "Compliance Officer": ["cag_tool"],
            "Stress Testing Engineer": ["code_execution"],
            "Model Validation Analyst": ["code_execution"],
            "Chief Risk Officer": [],
        }

        tool_map = {}
        for sid, agent in agent_map.items():
            role = agent.get("role", "")
            tool_map[sid] = role_tools.get(role, [])

        return tool_map
