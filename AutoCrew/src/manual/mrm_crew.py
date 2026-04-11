from crewai import Agent, Task, Crew, Process
from src.tools.code_execution import code_execution_tool


class ManualMRMCrew:
    """
    PATH A: Hardcoded MRM Crew from the base paper.
    Agents, tasks, and tools are manually defined in Python code.
    This is kept unchanged for comparison with AutoCrew.
    """

    def __init__(self, llm):
        self.llm = llm

    def run(self, modeling_output):
        # === AGENTS (hardcoded) ===

        compliance = Agent(
            role="Compliance Officer",
            goal="Check if documentation is complete.",
            backstory="You verify if Accuracy metrics were reported.",
            llm=self.llm,
            verbose=True,
        )

        stress_tester = Agent(
            role="Stress Testing Engineer",
            goal="Verify model robustness.",
            backstory="You write code to multiply inputs by 1.5 and check if the model still runs.",
            tools=[code_execution_tool],
            llm=self.llm,
            verbose=True,
        )

        judge = Agent(
            role="Chief Risk Officer",
            goal="Final Approval.",
            backstory="You review the stress test results and issue a VERDICT.",
            llm=self.llm,
            verbose=True,
        )

        # === TASKS (hardcoded) ===

        task1 = Task(
            description=f"Review this Model Card:\n{modeling_output}\nIs it complete?",
            expected_output="Compliance Check Result.",
            agent=compliance,
        )

        task2 = Task(
            description=(
                "Write and execute Python code to load the data, "
                "multiply numeric cols by 1.5, and print 'Stress Test Passed' if it works."
            ),
            expected_output="Stress Test Logs.",
            agent=stress_tester,
        )

        task3 = Task(
            description="Based on the stress test logs, is the model APPROVED or REJECTED?",
            expected_output="Final Verdict.",
            agent=judge,
            context=[task2],
        )

        # === CREW ===

        crew = Crew(
            agents=[compliance, stress_tester, judge],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

        return crew.kickoff()
