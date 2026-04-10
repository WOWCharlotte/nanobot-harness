"""Nanobot Harbor Agent — implements BaseInstalledAgent for Terminal Bench."""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)


class NanobotAgent(BaseInstalledAgent):
    """Nanobot agent integrated with Harbor/Terminal Bench."""

    SUPPORTS_ATIF = True

    # Nanobot tools to expose in benchmark context
    ALLOWED_TOOLS = [
        "Bash",
        "Read",
        "Write",
        "Glob",
        "Grep",
        "WebFetch",
        "WebSearch",
    ]

    @staticmethod
    def name() -> str:
        return AgentName.NANOBOT.value  # type: ignore

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "install-nanobot.sh.j2"

    # ------------------------------------------------------------------
    # Log parsing helpers (used by populate_context_post_run)
    # ------------------------------------------------------------------

    _LLM_USAGE_RE = re.compile(
        r"LLM usage: prompt=(\d+) completion=(\d+) cached=(\d+)"
    )
    _TOOL_CALL_RE = re.compile(
        r"Tool call: (\w+)\((.*)\)$"
    )

    @classmethod
    def _parse_llm_usage_line(cls, line: str) -> dict[str, int] | None:
        """Extract LLM usage from a debug log line.

        Matches: "LLM usage: prompt=1234 completion=567 cached=89"
        Returns: {"prompt_tokens": 1234, "completion_tokens": 567, "cached_tokens": 89}
        Returns None if no match.
        """
        m = cls._LLM_USAGE_RE.search(line)
        if not m:
            return None
        return {
            "prompt_tokens": int(m.group(1)),
            "completion_tokens": int(m.group(2)),
            "cached_tokens": int(m.group(3)),
        }

    @classmethod
    def _parse_tool_call_line(cls, line: str) -> dict[str, Any] | None:
        """Extract tool name and JSON args from a tool call log line.

        Matches: 'Tool call: Read({"file_path": "README.md"})'
        Returns: {"tool_name": "Read", "arguments": {"file_path": "README.md"}}
        Returns None if no match.
        """
        m = cls._TOOL_CALL_RE.search(line)
        if not m:
            return None
        tool_name = m.group(1)
        args_str = m.group(2).strip()
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = {"_raw": args_str}
        return {"tool_name": tool_name, "arguments": arguments}

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """Return the command to run nanobot agent with the given instruction."""
        escaped_instruction = shlex.quote(instruction)

        env: dict[str, str] = {
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY") or "",
            "NANOBOT_CONFIG_DIR": "/installed-agent",
            "PYTHONUNBUFFERED": "1",
            "NANOBOT_LOG_LEVEL": "DEBUG",  # Enable debug logs for token extraction
        }

        # Allow custom base URL for OpenRouter/self-hosted
        if os.environ.get("ANTHROPIC_BASE_URL"):
            env["ANTHROPIC_BASE_URL"] = os.environ["ANTHROPIC_BASE_URL"]

        config_path = "/installed-agent/nanobot_config.json"

        return [
            ExecInput(
                command=(
                    f"mkdir -p /installed-agent && "
                    f"nanobot agent -m {escaped_instruction} "
                    f"--config {config_path} "
                    f"--logs "
                    f"2>&1"
                ),
                env=env,
            )
        ]

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Parse nanobot stdout/stderr to build trajectory and extract metrics."""
        command_dir = self.logs_dir / "command-0"
        stdout_file = command_dir / "stdout.txt"
        stderr_file = command_dir / "stderr.txt"

        total_prompt = 0
        total_completion = 0
        total_cached = 0
        steps: list[Step] = []
        step_id = 1

        # Collect all log lines from both stdout and stderr
        log_lines: list[str] = []
        if stdout_file.exists():
            log_lines.extend(stdout_file.read_text(encoding="utf-8").splitlines())
        if stderr_file.exists():
            log_lines.extend(stderr_file.read_text(encoding="utf-8").splitlines())

        for line in log_lines:
            # Try LLM usage line
            usage = self._parse_llm_usage_line(line)
            if usage:
                total_prompt += usage["prompt_tokens"]
                total_completion += usage["completion_tokens"]
                total_cached += usage["cached_tokens"]
                continue

            # Try tool call line
            tool_info = self._parse_tool_call_line(line)
            if tool_info:
                call_id = f"call-{step_id}"
                tc = ToolCall(
                    tool_call_id=call_id,
                    function_name=tool_info["tool_name"],
                    arguments=tool_info["arguments"],
                )
                obs = Observation(
                    results=[
                        ObservationResult(
                            source_call_id=call_id,
                            content=None,
                            subagent_trajectory_ref=None,
                        )
                    ]
                )
                step = Step(
                    step_id=step_id,
                    source="agent",
                    message=f"Tool call: {tool_info['tool_name']}",
                    tool_calls=[tc],
                    observation=obs,
                )
                steps.append(step)
                step_id += 1

        # Build FinalMetrics
        metrics = None
        if total_prompt or total_completion or total_cached:
            metrics = FinalMetrics(
                total_prompt_tokens=total_prompt or None,
                total_completion_tokens=total_completion or None,
                total_cached_tokens=total_cached or None,
                total_cost_usd=None,
                total_steps=len(steps) or None,
                extra=None,
            )

        # Build trajectory
        trajectory = Trajectory(
            schema_version="ATIF-v1.2",
            session_id="nanobot-benchmark",
            agent=Agent(
                name=AgentName.NANOBOT.value,
                version=None,
                model_name=os.environ.get("NANOBOT_MODEL"),
                extra=None,
            ),
            steps=steps,
            final_metrics=metrics,
        )

        # Write trajectory
        trajectory_path = self.logs_dir / "trajectory.json"
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(trajectory.to_json_dict(), f, indent=2, ensure_ascii=False)

        # Populate context
        context.n_input_tokens = total_prompt or 0
        context.n_cache_tokens = total_cached or 0
        context.n_output_tokens = total_completion or 0
