import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Shared state passed between Modeling Crew and MRM Crew."""
    # Conversation history
    messages: Annotated[List[BaseMessage], operator.add]

    # Context
    dataset_path: str
    task_description: str

    # Artifacts passed from Modeling -> MRM
    modeling_report: str
    code_snippet: str
    mrm_verdict: str
