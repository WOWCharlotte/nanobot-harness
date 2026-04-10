"""Tests for NanobotAgent log parsing and token extraction."""

import pytest
from nanobot.harbor_agent.nanobot_agent import NanobotAgent


class TestParseLlmUsage:
    """Parse LLM usage from nanobot debug logs."""

    def test_parses_prompt_completion_cached_tokens(self):
        line = "LLM usage: prompt=1234 completion=567 cached=89"
        usage = NanobotAgent._parse_llm_usage_line(line)
        assert usage == {
            "prompt_tokens": 1234,
            "completion_tokens": 567,
            "cached_tokens": 89,
        }

    def test_parses_with_no_cached(self):
        line = "LLM usage: prompt=100 completion=50 cached=0"
        usage = NanobotAgent._parse_llm_usage_line(line)
        assert usage == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 0,
        }

    def test_returns_none_for_unmatched_line(self):
        line = 'Tool call: Read({"file_path": "README.md"})'
        usage = NanobotAgent._parse_llm_usage_line(line)
        assert usage is None

    def test_returns_none_for_empty_line(self):
        usage = NanobotAgent._parse_llm_usage_line("")
        assert usage is None


class TestParseToolCall:
    """Parse tool call lines from nanobot info logs."""

    def test_parses_tool_call(self):
        line = 'Tool call: Read({"file_path": "README.md"})'
        parsed = NanobotAgent._parse_tool_call_line(line)
        assert parsed == {
            "tool_name": "Read",
            "arguments": {"file_path": "README.md"},
        }

    def test_parses_tool_call_with_multiple_args(self):
        line = 'Tool call: Bash({"command": "ls -la", "cwd": "/tmp"})'
        parsed = NanobotAgent._parse_tool_call_line(line)
        assert parsed == {
            "tool_name": "Bash",
            "arguments": {"command": "ls -la", "cwd": "/tmp"},
        }

    def test_parses_tool_call_with_no_args(self):
        line = 'Tool call: Glob({"pattern": "*.py"})'
        parsed = NanobotAgent._parse_tool_call_line(line)
        assert parsed["tool_name"] == "Glob"
        assert parsed["arguments"] == {"pattern": "*.py"}

    def test_returns_none_for_non_tool_line(self):
        line = "LLM usage: prompt=100 completion=50 cached=0"
        parsed = NanobotAgent._parse_tool_call_line(line)
        assert parsed is None

    def test_returns_none_for_empty_line(self):
        parsed = NanobotAgent._parse_tool_call_line("")
        assert parsed is None
