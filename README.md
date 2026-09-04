---
title: OmniAgent - Autonomous AI Agent
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# 🤖 OmniAgent: Autonomous General-Purpose AI Agent

OmniAgent is a stateful, autonomous AI agent application built in Python using **LangGraph**. Designed as a versatile "do-everything" agent, it decomposes complex goals into dynamic multi-step plans, executes them with a comprehensive tool suite (web search, web scraping, code execution, shell commands, and sandboxed file management), self-reflects on results, and recovers from errors.

---

## 🌟 Key Features

- **Plan-and-Execute Architecture**: Breaks high-level objectives into sequential steps before execution.
- **Stateful Cyclic Execution (LangGraph)**: Maintains full context, tool outputs, step states, and checkpoint history.
- **Self-Critique & Verification**: Evaluates each step's outcome against the original goal and replans when errors occur.
- **Sandboxed Tool Suite**:
  - 🌐 **Live Web Search & Scraping**: DuckDuckGo search + HTML page text extractor.
  - 💻 **Python Code Execution**: In-process sandbox for calculations, data analysis, and script verification.
  - 📁 **Filesystem Management**: Sandboxed read, write, directory listing, and glob search.
  - 🖥️ **Shell Command Execution**: Safe bash execution inside the workspace directory.
- **Multi-Model Provider Support**:
  - **Google Gemini** (`gemini-2.5-flash`, `gemini-1.5-pro`)
  - **OpenAI** (`gpt-4o`, `gpt-4o-mini`)
  - **Anthropic** (`claude-3-5-sonnet`)
  - **Ollama / Local LLMs** (`llama3.2`, `deepseek-r1`, `qwen2.5`) with zero API keys.
- **Rich Terminal UI**: Streaming progress tables, colored tool call displays, and interactive REPL.

---

## 📁 Project Structure

```
omni-agent/
├── omni_agent/
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Settings and environment loader
│   ├── models.py           # Multi-provider LLM factory
│   ├── state.py            # LangGraph TypedDict state schema
│   ├── prompts.py          # Planner, Executor, Verifier & Synthesizer prompts
│   ├── graph.py            # LangGraph workflow engine
│   ├── cli.py              # Rich interactive REPL & CLI runner
│   └── tools/              # Agent tools suite
│       ├── __init__.py     # Tool registry & aggregator
│       ├── filesystem.py   # Read, write, list, search files
│       ├── code_runner.py  # Python code sandbox
│       ├── shell.py        # Safe bash executor
│       └── web.py          # Web search and page fetcher
├── tests/
│   ├── test_tools.py       # Unit tests for tools & sandbox security
│   └── test_agent.py       # Integration tests for state graph
├── workspace/              # Sandboxed folder for agent file creations
├── main.py                 # CLI entrypoint
├── requirements.txt        # Dependencies
├── pyproject.toml          # Project configuration
└── .env.example            # Environment variables template
```

---

## 🚀 Quickstart Guide

### 1. Set Up Virtual Environment

```bash
cd /Users/apple/.gemini/antigravity/scratch/omni-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the template and add your preferred provider's API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
DEFAULT_PROVIDER=gemini
DEFAULT_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your_gemini_key_here
```
*(Or set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or run local Ollama)*

### 3. Launch OmniAgent

**Interactive REPL Mode:**
```bash
python main.py
```

**One-Shot Command Mode:**
```bash
python main.py "Search for recent AI agent benchmarks, write a Python script to verify matrix multiplication performance, and save the findings to report.md"
```

**Switch Model / Provider on the Fly:**
```bash
python main.py --provider openai --model gpt-4o "Analyze my workspace data"
python main.py --provider ollama --model deepseek-r1:8b "Plan a software architecture"
```

---

## 🛠️ Interactive REPL Commands

While inside the interactive session:
- `/tools` - View all registered tools and their descriptions.
- `/reset` - Clear conversation thread memory and start fresh.
- `/help` - Show available commands.
- `/exit` - Exit the agent CLI.

---

## 🔌 Adding Custom Tools

Adding custom tools to OmniAgent is easy using LangChain's `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def calculate_mortgage(principal: float, rate_annual: float, years: int) -> str:
    """Calculate monthly mortgage payment given principal, rate, and term."""
    r = rate_annual / 100 / 12
    n = years * 12
    payment = principal * (r * (1 + r)**n) / ((1 + r)**n - 1)
    return f"Monthly Payment: ${payment:.2f}"
```

Then register it in `omni_agent/tools/__init__.py` inside `get_default_tools()`.
