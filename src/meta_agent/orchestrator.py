from src.meta_agent.task_decomposer import TaskDecomposer
from src.meta_agent.agent_selector import AgentSelector
from src.meta_agent.tool_assigner import ToolAssigner
from src.meta_agent.instruction_writer import InstructionWriter
from src.meta_agent.json_generator import JSONGenerator


class MetaAgentOrchestrator:
    """
    The brain of AutoCrew.
    Chains the 5-step meta agent pipeline:
      1. Task Decomposition
      2. Agent Selection
      3. Tool Assignment
      4. Instruction Writing
      5. JSON Config Generation
      
    Input:  Natural language task + dataset path
    Output: Complete crew_config.json ready for CrewAI execution
    """

    def __init__(self, llm, llm_provider_name: str = "ollama"):
        self.llm = llm
        self.llm_provider_name = llm_provider_name

        # Initialize all pipeline steps
        self.decomposer = TaskDecomposer(llm)
        self.agent_selector = AgentSelector(llm)
        self.tool_assigner = ToolAssigner(llm)
        self.instruction_writer = InstructionWriter(llm)
        self.json_generator = JSONGenerator()

    def run(self, task_description: str, dataset_path: str) -> dict:
        """
        Execute the full meta agent pipeline.
        
        Args:
            task_description: e.g. "Predict credit approval (Binary Classification)"
            dataset_path: e.g. "data/credit_card_approval.csv"
            
        Returns:
            Complete crew configuration dict
        """
        print("\n" + "=" * 50)
        print("META AGENT: Generating Crew Configuration")
        print("=" * 50)
        print(f"Task: {task_description}")
        print(f"Dataset: {dataset_path}")
        print(f"LLM: {self.llm_provider_name}")
        print()

        # Step 1: Decompose task into subtasks
        subtasks = self.decomposer.decompose(task_description, dataset_path)

        # Cap subtasks to avoid rate limits on free tier APIs
        if len(subtasks.get("modeling_subtasks", [])) > 4:
            print(f"[Meta Agent] Capping modeling subtasks from {len(subtasks['modeling_subtasks'])} to 4")
            subtasks["modeling_subtasks"] = subtasks["modeling_subtasks"][:4]
        if len(subtasks.get("mrm_subtasks", [])) > 3:
            print(f"[Meta Agent] Capping MRM subtasks from {len(subtasks['mrm_subtasks'])} to 3")
            subtasks["mrm_subtasks"] = subtasks["mrm_subtasks"][:3]

        # Step 2: Select agents for each subtask
        agent_map = self.agent_selector.select_agents(subtasks)

        # Step 3: Assign tools to each agent
        tool_map = self.tool_assigner.assign_tools(subtasks, agent_map)

        # Step 4: Write detailed instructions
        instructions = self.instruction_writer.write_instructions(
            subtasks, agent_map, tool_map, dataset_path, task_description
        )

        # Step 5: Generate the final JSON config
        config = self.json_generator.generate(
            task_description=task_description,
            dataset_path=dataset_path,
            llm_provider=self.llm_provider_name,
            subtasks=subtasks,
            agent_map=agent_map,
            tool_map=tool_map,
            instructions=instructions,
        )

        print("\n" + "=" * 50)
        print("META AGENT: Configuration Generated Successfully!")
        print("=" * 50)

        self._print_summary(config)

        return config

    def _print_summary(self, config: dict):
        """Print a readable summary of the generated config."""
        print("\n--- Generated Crew Summary ---\n")

        for crew_name in ["modeling_crew", "mrm_crew"]:
            crew = config.get(crew_name, {})
            agents = crew.get("agents", [])
            tasks = crew.get("tasks", [])

            label = crew_name.replace("_", " ").title()
            print(f"{label}:")
            print(f"  Agents: {len(agents)}")
            for a in agents:
                tools_str = ", ".join(a["tools"]) if a["tools"] else "none"
                print(f"    - {a['role']} (tools: {tools_str})")
            print(f"  Tasks: {len(tasks)}")
            for i, t in enumerate(tasks):
                desc_short = t["description"][:80] + "..." if len(t["description"]) > 80 else t["description"]
                print(f"    {i+1}. {desc_short}")
            print()
