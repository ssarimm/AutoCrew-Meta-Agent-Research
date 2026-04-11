from crewai.tools import BaseTool


class CAGTool(BaseTool):
    name: str = "cache_augmented_generation"
    description: str = (
        "Compares modeling documentation against an organizational modeling guide. "
        "Input should be a string containing the modeling crew's documentation text. "
        "Returns compliance check results."
    )

    def _run(self, documentation: str) -> str:
        """
        In the base paper, CAG loads a PDF modeling guide and compares.
        For our implementation, we do a simplified text-based compliance check.
        The LLM agent will use its reasoning to compare the documentation
        against standard ML pipeline steps.
        """
        required_steps = [
            "data extraction",
            "exploratory data analysis",
            "feature engineering",
            "model selection",
            "hyperparameter tuning",
            "model training",
            "model evaluation",
        ]

        doc_lower = documentation.lower()
        found = []
        missing = []

        for step in required_steps:
            if step in doc_lower:
                found.append(step)
            else:
                missing.append(step)

        report = "COMPLIANCE CHECK REPORT\n"
        report += "=" * 40 + "\n\n"

        report += f"Steps Found ({len(found)}/{len(required_steps)}):\n"
        for s in found:
            report += f"  [PASS] {s.title()}\n"

        if missing:
            report += f"\nSteps Missing ({len(missing)}):\n"
            for s in missing:
                report += f"  [FAIL] {s.title()}\n"
        else:
            report += "\nAll required steps are documented.\n"

        report += f"\nCompliance Score: {len(found)}/{len(required_steps)}"
        report += f" ({round(len(found)/len(required_steps)*100, 1)}%)\n"

        return report


# Singleton
cag_tool = CAGTool()
