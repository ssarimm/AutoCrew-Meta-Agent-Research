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

        # CRITICAL: Ensure model training is always present in modeling subtasks.
        modeling_tasks = subtasks.get("modeling_subtasks", [])
        training_kw = ["train", "classifier", "fit model", "model training", "model evaluation", "evaluat"]
        has_training = any(
            any(kw in (st.get("name", "") + " " + st.get("description", "")).lower()
                for kw in training_kw)
            for st in modeling_tasks
        )
        if not has_training and modeling_tasks:
            inject_idx = max(0, len(modeling_tasks) - 1)  # insert before last (docs) task
            prev_id = modeling_tasks[inject_idx - 1]["id"] if inject_idx > 0 else None
            training_task = {
                "id": "task_train_injected",
                "name": "Model Training and Evaluation",
                "description": (
                    f"Train a RandomForestClassifier on '{dataset_path}'. "
                    f"Load data, handle NaN, encode categoricals, drop unnamed index columns. "
                    f"Identify target column from task: {task_description}. "
                    f"Split 80/20. Train RandomForestClassifier(n_estimators=100, random_state=42). "
                    f"Print EXACTLY: Accuracy: X.XXXX, F1 Score: X.XXXX, Precision: X.XXXX, Recall: X.XXXX"
                ),
                "depends_on": [prev_id] if prev_id else [],
            }
            modeling_tasks.insert(inject_idx, training_task)  # insert before docs, not replace
            subtasks["modeling_subtasks"] = modeling_tasks
            print(f"[Meta Agent] No training task found — injected 'Model Training and Evaluation' at slot {inject_idx}")

        # Step 2: Select agents for each subtask
        agent_map = self.agent_selector.select_agents(subtasks)

        # Step 3: Assign tools to each agent
        tool_map = self.tool_assigner.assign_tools(subtasks, agent_map)

        # Step 4: Write detailed instructions
        instructions = self.instruction_writer.write_instructions(
            subtasks, agent_map, tool_map, dataset_path, task_description
        )

        # Post-process: inject dataset path + target column into every code-executing task.
        target_hint = self._extract_target_hint(task_description, dataset_path)

        for sid, instr in instructions.items():
            desc = instr.get("task_description", "")
            tool_list_str = str(tool_map.get(sid, []))
            has_code_tool = "code_execution" in tool_list_str or "execute_python_code" in tool_list_str
            if has_code_tool:
                suffix = ""
                if dataset_path not in desc:
                    suffix += f"\n\nIMPORTANT: Load the real dataset from '{dataset_path}'. Do NOT generate fake data."
                if target_hint and target_hint not in desc:
                    suffix += f"\n\n{target_hint}"

                # Inject dataset-specific loading hints
                ds_hint = self._get_dataset_hint(dataset_path)
                if ds_hint and ds_hint not in desc:
                    suffix += f"\n\n{ds_hint}"

                is_eda_or_load = any(kw in desc.lower() for kw in [
                    "exploratory", "eda", "summary statistics", "data types", "missing values",
                    "print df.shape", "print(df.shape", "first 5 rows", "df.head()"
                ])
                is_training_task = any(kw in desc.lower() for kw in [
                    "train", "split", "randomforest", "classifier", "model", "evaluat", "stress"
                ])
                if not target_hint and "known = " not in desc and "df.columns[-1]" not in desc:
                    if is_training_task and not is_eda_or_load:
                        suffix += (
                            f"\n\nCRITICAL: Do NOT use column names from previous task outputs — "
                            f"prior agents may have hallucinated renamed columns. "
                            f"Always detect the target with: "
                            f"known = {{'SeriousDlqin2yrs', 'Class'}}; "
                            f"target_col = next((c for c in known if c in df.columns), df.columns[-1]); "
                            f"X = df.drop(columns=[target_col] + [c for c in df.columns if 'unnamed' in c.lower()]); "
                            f"y = df[target_col]"
                        )

                # Always inject the weighted average reminder for training/eval tasks
                if is_training_task and "average='weighted'" not in desc:
                    suffix += (
                        "\n\nCRITICAL: For f1_score, precision_score, and recall_score, "
                        "ALWAYS use average='weighted'. NEVER use the default binary average."
                    )

                if suffix:
                    instr["task_description"] = desc + suffix

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

    def _get_dataset_hint(self, dataset_path: str) -> str:
        """Return dataset-specific loading hints for the LLM agents."""
        if "credit_card_approval" in dataset_path:
            return (
                "DATASET HINT: 'credit_card_approval.csv' has NO header row. "
                "Use pd.read_csv('...', header=None). Columns are integers 0-15. "
                "Target is the last column (df.columns[-1]). "
                "Replace '?' with NaN, then dropna()."
            )
        elif "cs-training" in dataset_path:
            return (
                "DATASET HINT: 'cs-training.csv' has ~30K missing values in MonthlyIncome. "
                "Use df.fillna(df.median(numeric_only=True)) instead of dropna(). "
                "Drop 'Unnamed: 0' (row index). Target is 'SeriousDlqin2yrs'."
            )
        elif "creditcard_2023" in dataset_path:
            return (
                "DATASET HINT: 'creditcard_2023.csv' is highly imbalanced (~0.17% fraud). "
                "Use class_weight='balanced' in RandomForestClassifier. "
                "Drop 'id' column with errors='ignore': df.drop(columns=['id'], errors='ignore'). "
                "Target is 'Class'."
            )
        return ""

    def _extract_target_hint(self, task_description: str, dataset_path: str = "") -> str:
        """
        Parse the NL task description and dataset path for an explicit target column.
        Returns a ready-to-inject code hint. Returns empty string if nothing is found.
        """
        import re

        # Dataset-path based detection (most reliable)
        if "credit_card_approval" in dataset_path:
            return (
                "TARGET COLUMN: Use df.columns[-1] (the last column). "
                "This CSV has no header — use pd.read_csv(..., header=None). "
                "X = df.drop(columns=[df.columns[-1]]); y = df[df.columns[-1]]"
            )
        elif "cs-training" in dataset_path:
            return (
                "TARGET COLUMN IS 'SeriousDlqin2yrs'. "
                "Drop 'Unnamed: 0' (it's a row index, not a feature). Use:\n"
                "  X = df.drop(columns=['SeriousDlqin2yrs'])\n"
                "  y = df['SeriousDlqin2yrs']"
            )
        elif "creditcard_2023" in dataset_path:
            return (
                "TARGET COLUMN IS 'Class'. "
                "Drop 'id' column if present. Use:\n"
                "  X = df.drop(columns=['Class'])\n"
                "  y = df['Class']"
            )

        # Try NL parsing as fallback
        m = re.search(r"[Tt]arget\s+column[:\s]+['\"]?([\w]+)['\"]?", task_description)
        if m:
            col = m.group(1)
            drop_unnamed = "[c for c in df.columns if 'unnamed' in c.lower()]"
            return (
                f"TARGET COLUMN IS '{col}' — do NOT use df.iloc[:, -1]. Use:\n"
                f"  target_col = '{col}'\n"
                f"  X = df.drop(columns=[target_col] + {drop_unnamed})\n"
                f"  y = df[target_col]"
            )

        m = re.search(r"y\s*=\s*df\[['\"](\w+)['\"]\]", task_description)
        if m:
            col = m.group(1)
            drop_unnamed = "[c for c in df.columns if 'unnamed' in c.lower()]"
            return (
                f"TARGET COLUMN IS '{col}' — do NOT use df.iloc[:, -1]. Use:\n"
                f"  target_col = '{col}'\n"
                f"  X = df.drop(columns=[target_col] + {drop_unnamed})\n"
                f"  y = df[target_col]"
            )

        return ""

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
