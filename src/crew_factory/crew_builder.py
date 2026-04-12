import json
from crewai import Agent, Task, Crew, Process
from src.tools.tool_registry import ToolRegistry


class CrewBuilder:
    """
    Parses a crew config JSON and builds live CrewAI Agent, Task, and Crew objects.
    This is the bridge between the Meta Agent's output and CrewAI execution.
    """

    def __init__(self, llm):
        self.llm = llm

    def build_crew(self, crew_config: dict) -> Crew:
        """
        Build a CrewAI Crew from a crew config dict.
        
        Args:
            crew_config: dict with "agents" and "tasks" lists
            
        Returns:
            CrewAI Crew object ready to kickoff()
        """
        # Build agents
        agents_by_id = {}
        agent_list = []

        for agent_def in crew_config.get("agents", []):
            agent = self._build_agent(agent_def)
            agents_by_id[agent_def["id"]] = agent
            agent_list.append(agent)

        # Build tasks
        task_list = []
        for task_def in crew_config.get("tasks", []):
            task = self._build_task(task_def, agents_by_id, task_list)
            task_list.append(task)

        # Build crew
        process_type = crew_config.get("process", "sequential")
        process = Process.sequential if process_type == "sequential" else Process.hierarchical

        crew = Crew(
            agents=agent_list,
            tasks=task_list,
            process=process,
            memory=False,
            verbose=False,  # disables Live status tree; agent panels still show via Agent.verbose
        )

        return crew

    def _build_agent(self, agent_def: dict) -> Agent:
        """Build a single CrewAI Agent from config."""
        # Resolve tools from registry
        tool_names = agent_def.get("tools", [])
        tools = ToolRegistry.get_tools(tool_names)

        agent = Agent(
            role=agent_def["role"],
            goal=agent_def["goal"],
            backstory=agent_def.get("backstory", "You are a skilled professional."),
            tools=tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=agent_def.get("allow_delegation", False),
            max_iter=8,
        )

        return agent

    def _build_task(self, task_def: dict, agents_by_id: dict, existing_tasks: list) -> Task:
        """Build a single CrewAI Task from config."""
        # Resolve agent
        agent_id = task_def["agent_id"]
        agent = agents_by_id.get(agent_id)
        if agent is None:
            raise ValueError(f"Agent '{agent_id}' not found. Available: {list(agents_by_id.keys())}")

        # Resolve context dependencies (tasks this task depends on)
        context = []
        for idx in task_def.get("context_from_indices", []):
            if 0 <= idx < len(existing_tasks):
                context.append(existing_tasks[idx])

        task_kwargs = {
            "description": task_def["description"],
            "expected_output": task_def.get("expected_output", "Task output."),
            "agent": agent,
        }

        if context:
            task_kwargs["context"] = context

        return Task(**task_kwargs)

    def build_from_full_config(self, full_config: dict) -> tuple:
        """
        Build both modeling and MRM crews from a full config.
        
        Args:
            full_config: Complete config with "modeling_crew" and "mrm_crew"
            
        Returns:
            (modeling_crew, mrm_crew) tuple of Crew objects
        """
        modeling_crew = None
        mrm_crew = None

        if "modeling_crew" in full_config:
            print("\n[Crew Builder] Building Modeling Crew...")
            modeling_crew = self.build_crew(full_config["modeling_crew"])
            n_agents = len(full_config["modeling_crew"].get("agents", []))
            n_tasks = len(full_config["modeling_crew"].get("tasks", []))
            print(f"[Crew Builder] Modeling Crew: {n_agents} agents, {n_tasks} tasks")

        if "mrm_crew" in full_config:
            print("\n[Crew Builder] Building MRM Crew...")
            mrm_crew = self.build_crew(full_config["mrm_crew"])
            n_agents = len(full_config["mrm_crew"].get("agents", []))
            n_tasks = len(full_config["mrm_crew"].get("tasks", []))
            print(f"[Crew Builder] MRM Crew: {n_agents} agents, {n_tasks} tasks")

        return modeling_crew, mrm_crew

    @staticmethod
    def load_config_from_file(filepath: str) -> dict:
        """Load a crew config from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
