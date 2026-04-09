from src.tools.code_execution import code_execution_tool
from src.tools.eda_tool import eda_tool
from src.tools.cag_tool import cag_tool


class ToolRegistry:
    """
    Central registry mapping tool name strings to actual tool instances.
    Used by the crew_builder to resolve tool names from JSON config.
    """

    _registry = {
        "code_execution": code_execution_tool,
        "execute_python_code": code_execution_tool,
        "eda_tool": eda_tool,
        "exploratory_data_analysis": eda_tool,
        "cag_tool": cag_tool,
        "cache_augmented_generation": cag_tool,
    }

    @classmethod
    def get_tool(cls, name: str):
        """Get a tool instance by name. Returns None if not found."""
        tool = cls._registry.get(name.lower().strip())
        if tool is None:
            print(f"WARNING: Tool '{name}' not found in registry.")
        return tool

    @classmethod
    def get_tools(cls, names: list) -> list:
        """Get multiple tool instances by name list. Skips unknown tools."""
        tools = []
        for name in names:
            tool = cls.get_tool(name)
            if tool is not None:
                tools.append(tool)
        return tools

    @classmethod
    def list_available(cls) -> list:
        """List all registered tool names."""
        return list(set(
            name for name, tool in cls._registry.items()
        ))

    @classmethod
    def get_catalog(cls) -> dict:
        """Get tool catalog with names and descriptions (for meta agent)."""
        seen = set()
        catalog = {}
        for name, tool in cls._registry.items():
            if tool.name not in seen:
                seen.add(tool.name)
                catalog[name] = {
                    "name": tool.name,
                    "description": tool.description
                }
        return catalog
