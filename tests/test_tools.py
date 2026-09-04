"""Unit tests for OmniAgent built-in tools."""

import unittest
import shutil
import tempfile
from pathlib import Path

from omni_agent.config import AgentConfig
from omni_agent.tools.filesystem import (
    read_file,
    write_file,
    list_directory,
    file_search,
    set_active_config as set_fs_config,
)
from omni_agent.tools.code_runner import (
    execute_python,
    set_active_config as set_code_config,
)
from omni_agent.tools.shell import (
    run_shell_command,
    set_active_config as set_shell_config,
)


class TestOmniAgentTools(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = AgentConfig(workspace_dir=self.test_dir)
        set_fs_config(self.config)
        set_code_config(self.config)
        set_shell_config(self.config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_filesystem_write_and_read(self):
        # Write
        result = write_file.invoke({"file_path": "notes/todo.txt", "content": "1. Build agent\n2. Conquer tests"})
        self.assertIn("Successfully wrote", result)
        self.assertTrue((self.test_dir / "notes/todo.txt").exists())

        # Read
        content = read_file.invoke({"file_path": "notes/todo.txt"})
        self.assertIn("1. Build agent", content)

    def test_filesystem_sandbox_security(self):
        # Attempt path traversal
        result = read_file.invoke({"file_path": "../../outside.txt"})
        self.assertIn("Access denied", result)

    def test_filesystem_list_and_search(self):
        write_file.invoke({"file_path": "data1.csv", "content": "a,b,c"})
        write_file.invoke({"file_path": "data2.csv", "content": "1,2,3"})

        listing = list_directory.invoke({"dir_path": "."})
        self.assertIn("data1.csv", listing)
        self.assertIn("data2.csv", listing)

        search_res = file_search.invoke({"pattern": "*.csv", "dir_path": "."})
        self.assertIn("data1.csv", search_res)

    def test_code_runner_execution(self):
        code = "a = 21\nb = 2\nprint(f'Result: {a * b}')"
        output = execute_python.invoke({"code": code})
        self.assertIn("Result: 42", output)
        self.assertIn("[Exit code: 0]", output)

    def test_shell_runner_execution(self):
        cmd = "echo 'OmniAgent shell works'"
        output = run_shell_command.invoke({"command": cmd})
        self.assertIn("OmniAgent shell works", output)
        self.assertIn("[Exit code: 0]", output)


if __name__ == "__main__":
    unittest.main()
