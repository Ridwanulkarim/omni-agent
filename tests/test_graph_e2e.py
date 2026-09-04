"""End-to-end test of OmniAgent StateGraph using a deterministic mock model."""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

from omni_agent.config import AgentConfig
from omni_agent.graph import create_omni_agent
import tempfile
import shutil
from pathlib import Path


class MockLLM(BaseChatModel):
    """Simple mock LLM returning pre-scripted responses."""
    responses: list = []
    call_index: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.call_index < len(self.responses):
            resp = self.responses[self.call_index]
            self.call_index += 1
        else:
            resp = AIMessage(content="Final summary: task complete.")
        return ChatResult(generations=[ChatGeneration(message=resp)])

    @property
    def _llm_type(self) -> str:
        return "mock"

    def bind_tools(self, tools, **kwargs):
        return self


class TestOmniAgentGraph(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = AgentConfig(workspace_dir=self.test_dir, max_steps=5)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_mock_agent_turn(self):
        mock_llm = MockLLM()
        # Scripted responses:
        # 1. Planner output (JSON plan)
        planner_msg = AIMessage(content='{"goal": "Write greeting", "steps": [{"step_id": 1, "description": "Write greeting file", "tool_hint": "write_file"}]}')
        # 2. Executor calls tool
        tool_call_msg = AIMessage(
            content="I will write a greeting file.",
            tool_calls=[{
                "id": "tc1",
                "name": "write_file",
                "args": {"file_path": "hello.txt", "content": "Hello OmniAgent!"}
            }]
        )
        # 3. Executor after tool response
        after_tool_msg = AIMessage(content="Greeting file written successfully.")
        # 4. Verifier output
        verifier_msg = AIMessage(content="COMPLETED: The greeting file was written.")
        # 5. Synthesizer output
        synthesizer_msg = AIMessage(content="All done! The greeting file is in the workspace.")

        mock_llm.responses = [planner_msg, tool_call_msg, after_tool_msg, verifier_msg, synthesizer_msg]

        with patch("omni_agent.graph.get_chat_model", return_value=mock_llm):
            app = create_omni_agent(self.config)
            
            initial_state = {
                "user_goal": "Write a greeting file",
                "messages": [HumanMessage(content="Write a greeting file")],
                "iteration": 0,
                "max_iterations": 5,
                "is_completed": False,
            }
            thread_config = {"configurable": {"thread_id": "test-thread-1"}}
            
            final_state = app.invoke(initial_state, thread_config)

            # Check that file was actually created by real tool execution
            greeting_file = self.test_dir / "hello.txt"
            self.assertTrue(greeting_file.exists())
            self.assertEqual(greeting_file.read_text(), "Hello OmniAgent!")

            # Check that final response is populated
            self.assertTrue(final_state["is_completed"])
            self.assertIn("All done!", final_state["final_response"])


if __name__ == "__main__":
    unittest.main()
