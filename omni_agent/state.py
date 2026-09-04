"""LangGraph state schema for OmniAgent."""

from typing import List, Optional, Dict, Any, Sequence
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PlanStep(BaseModel):
    """A discrete step in the agent's execution plan."""
    step_id: int
    description: str
    tool_hint: Optional[str] = None
    status: str = Field(default="pending", description="pending | in_progress | completed | failed")
    result: Optional[str] = None


class ExecutionPlan(BaseModel):
    """Structured plan composed of steps."""
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)


class AgentState(TypedDict):
    """The central state of the OmniAgent execution graph."""
    
    # Message stream between user, agent, and tools
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Original user goal
    user_goal: str
    
    # Structured multi-step plan
    plan: Optional[ExecutionPlan]
    current_step_index: int
    
    # Execution metrics and guardrails
    iteration: int
    max_iterations: int
    
    # Reflection & Verification
    is_completed: bool
    verification_feedback: Optional[str]
    
    # Final synthesized answer
    final_response: Optional[str]
