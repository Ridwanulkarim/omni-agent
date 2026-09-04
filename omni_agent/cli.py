import warnings
warnings.filterwarnings("ignore")
"""Interactive Command-Line Interface for OmniAgent."""

import argparse
import sys
import uuid
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.rule import Rule
from rich.prompt import Prompt

from langchain_core.messages import HumanMessage
from omni_agent.config import AgentConfig
from omni_agent.graph import create_omni_agent
from omni_agent.tools import get_default_tools

console = Console()


def print_banner(config: AgentConfig):
    """Display the OmniAgent welcome banner."""
    welcome_text = (
        f"[bold cyan]OmniAgent[/bold cyan] — [bold green]Autonomous General-Purpose AI Agent[/bold green]\n\n"
        f"• [dim]Provider:[/dim] [yellow]{config.provider}[/yellow]  "
        f"• [dim]Model:[/dim] [yellow]{config.model}[/yellow]\n"
        f"• [dim]Workspace:[/dim] [blue]{config.workspace_dir}[/blue]  "
        f"• [dim]Max Steps:[/dim] [magenta]{config.max_steps}[/magenta]\n\n"
        f"[dim]Type your prompt to run, or use commands: [/dim]"
        f"[bold]/tools[/bold], [bold]/help[/bold], [bold]/reset[/bold], [bold]/exit[/bold]"
    )
    console.print(Panel(welcome_text, title="🤖 OmniAgent", border_style="cyan"))


def list_tools_command(config: AgentConfig):
    """Show table of all available tools."""
    tools = get_default_tools(config)
    table = Table(title="🛠️ Registered Agent Tools", border_style="blue")
    table.add_column("Tool Name", style="bold cyan", width=20)
    table.add_column("Description", style="dim", overflow="fold")

    for t in tools:
        table.add_row(t.name, t.description.strip().split("\n")[0])

    console.print(table)


def run_agent_turn(app, prompt: str, thread_id: str, config: AgentConfig):
    """Execute a single agent turn and stream progress events to the console."""
    console.print(Rule(style="dim cyan"))
    console.print(f"[bold green]User Goal:[/bold green] {prompt}")
    console.print(Rule(style="dim cyan"))

    initial_state = {
        "user_goal": prompt,
        "messages": [HumanMessage(content=prompt)],
        "iteration": 0,
        "max_iterations": config.max_steps,
        "is_completed": False,
    }

    thread_config = {"configurable": {"thread_id": thread_id}}

    with console.status("[bold cyan]OmniAgent is thinking & planning...", spinner="dots"):
        for event in app.stream(initial_state, thread_config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "planner":
                    plan = node_output.get("plan")
                    if plan:
                        plan_table = Table(title="📋 Generated Execution Plan", border_style="cyan")
                        plan_table.add_column("#", width=4, justify="center")
                        plan_table.add_column("Step Description", style="bold")
                        plan_table.add_column("Status", style="yellow", justify="center")

                        for step in plan.steps:
                            plan_table.add_row(
                                str(step.step_id), step.description, step.status
                            )
                        console.print(plan_table)

                elif node_name == "executor":
                    messages = node_output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                                console.print(
                                    f"[bold yellow]⚙️ Calling Tool:[/bold yellow] "
                                    f"[cyan]{tc['name']}[/cyan]({args_str[:120]}{'...' if len(args_str) > 120 else ''})"
                                )
                        elif last_msg.content:
                            console.print(f"[dim]Agent thought:[/dim] {last_msg.content[:200]}...")

                elif node_name == "tools":
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        content_preview = str(msg.content).strip()
                        if len(content_preview) > 300:
                            content_preview = content_preview[:300] + "..."
                        console.print(
                            f"[dim green]✓ Tool Output:[/dim green] [dim]{content_preview}[/dim]"
                        )

                elif node_name == "verifier":
                    feedback = node_output.get("verification_feedback")
                    if feedback:
                        console.print(f"[bold magenta]🔍 Verifier Feedback:[/bold magenta] {feedback}")
                    elif node_output.get("is_completed"):
                        console.print("[bold green]✓ Verifier: Step/Goal Completed![/bold green]")

                elif node_name == "synthesizer":
                    final_response = node_output.get("final_response", "")
                    console.print("\n")
                    console.print(
                        Panel(
                            Markdown(final_response),
                            title="✨ Final Result",
                            border_style="green",
                            padding=(1, 2),
                        )
                    )


def main():
    parser = argparse.ArgumentParser(description="OmniAgent: Autonomous General-Purpose AI Agent")
    parser.add_argument("query", nargs="?", default=None, help="One-shot prompt to execute")
    parser.add_argument("--provider", default=None, help="LLM provider: gemini, openai, anthropic, ollama")
    parser.add_argument("--model", default=None, help="Model name (e.g. gemini-2.5-flash, gpt-4o)")
    parser.add_argument("--workspace", default=None, help="Custom path to workspace directory")
    parser.add_argument("--max-steps", type=int, default=None, help="Max execution steps per goal")

    args = parser.parse_args()

    # Load config with CLI overrides
    config = AgentConfig()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.workspace:
        config.workspace_dir = Path(args.workspace).resolve()
    if args.max_steps:
        config.max_steps = args.max_steps

    print_banner(config)

    try:
        app = create_omni_agent(config)
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {str(e)}")
        sys.exit(1)

    thread_id = str(uuid.uuid4())

    # One-shot mode
    if args.query:
        run_agent_turn(app, args.query, thread_id, config)
        return

    # Interactive REPL mode
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]OmniAgent[/bold cyan]")
            if not user_input.strip():
                continue

            clean_cmd = user_input.strip().lower()
            if clean_cmd in ("/exit", "exit", "quit", ":q"):
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            elif clean_cmd == "/tools":
                list_tools_command(config)
                continue
            elif clean_cmd == "/help":
                console.print(
                    "[bold]Available Commands:[/bold]\n"
                    "  [cyan]/tools[/cyan]    - List all registered agent tools\n"
                    "  [cyan]/reset[/cyan]    - Clear agent conversation memory\n"
                    "  [cyan]/help[/cyan]     - Show this help message\n"
                    "  [cyan]/exit[/cyan]     - Exit the session"
                )
                continue
            elif clean_cmd == "/reset":
                thread_id = str(uuid.uuid4())
                console.print("[bold yellow]Conversation thread reset.[/bold yellow]")
                continue

            run_agent_turn(app, user_input, thread_id, config)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Exiting OmniAgent. Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Execution Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
