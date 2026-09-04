"""Integration and unit tests for OmniAgent graph."""

import unittest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from omni_agent.config import AgentConfig
from omni_agent.state import ExecutionPlan, PlanStep, AgentState


class TestOmniAgentState(unittest.TestCase):

    def test_plan_structure(self):
        plan = ExecutionPlan(
            goal="Test user goal",
            steps=[
                PlanStep(step_id=1, description="Step 1", tool_hint="web_search"),
                PlanStep(step_id=2, description="Step 2", tool_hint="write_file"),
            ]
        )
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].status, "pending")
        self.assertEqual(plan.steps[1].tool_hint, "write_file")


if __name__ == "__main__":
    unittest.main()
