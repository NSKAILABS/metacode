from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from mcp_server.tools import TOOLS, dispatch

from llm.groq_client import get_client, get_model_config, load_env


SYSTEM_PROMPT = """\
You are a careful meta-optics design assistant. You have access to a
small toolbox of *verified* functions backed by a physics simulator
(metabox3 RCWA). Never invent parameters — start from a template.

The correct procedure for any design request is:

  1. Call `list_templates` to see verified starting points.
  2. Call `get_template` on the most relevant one.
  3. If you need help, call `get_workflow_guide`.
  4. Adapt the template's `params` to the user's request.
  5. Call `validate_design(params=...)`. If invalid, fix the reported
     errors and validate again.
  6. Call `run_simulation(params=..., n_samples=20)`.
  7. If phase_coverage < 0.9 OR mean_transmission < 0.7, call
     `get_optimization_tips(result=...)` and adjust, then re-simulate.

Cap yourself at 5 tool calls total. Once the metrics are acceptable OR
you have applied one round of tips, stop and report the final design
parameters and metrics to the user in a short, structured summary.
Do not run more simulations than needed.
"""


@dataclass
class AgentResult:
    final_text: str
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stopped_on: str = "final_text"  # or "iteration_cap"


def _extract_tool_calls(msg) -> list[Any]:
    """Groq/OpenAI response shape: .tool_calls or None."""
    tc = getattr(msg, "tool_calls", None) or []
    return list(tc)


def run_agent(user_prompt: str, max_iterations: int | None = None) -> AgentResult:
    """Drive the tool-calling loop for a single user request."""
    load_env()
    if max_iterations is None:
        max_iterations = int(os.environ.get("AGENT_MAX_ITERATIONS", "5"))
    max_iterations = max(1, min(max_iterations, 5))

    client = get_client()
    cfg = get_model_config()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    tool_call_log: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        resp = client.chat.completions.create(
            model=cfg["model"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            tools=TOOLS,
            tool_choice="auto",
            messages=messages,
        )
        msg = resp.choices[0].message
        calls = _extract_tool_calls(msg)

        # Append the assistant turn (with any tool_calls) verbatim.
        assistant_turn: dict[str, Any] = {"role": "assistant",
                                          "content": msg.content or ""}
        if calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments,
                    },
                }
                for c in calls
            ]
        messages.append(assistant_turn)

        # No tool calls -> the LLM produced a final answer.
        if not calls:
            return AgentResult(
                final_text=msg.content or "",
                iterations=iteration,
                tool_calls=tool_call_log,
                stopped_on="final_text",
            )

        # Execute each requested tool and append the tool result turn.
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError as e:
                args = {}
                result = {"ok": False,
                          "error": f"Malformed JSON arguments: {e}"}
            else:
                result = dispatch(name, args)

            # Full arrays go into the log for the caller, but the LLM only
            # sees the trimmed summary.
            log_entry = {"iteration": iteration, "name": name, "args": args,
                         "result": result}
            tool_call_log.append(log_entry)

            llm_visible = _strip_full_arrays(result)
            messages.append({
                "role": "tool",
                "tool_call_id": c.id,
                "name": name,
                "content": json.dumps(llm_visible, default=str)[:6000],
            })

    # Fell out of the loop without a final text response.
    return AgentResult(
        final_text="[Agent hit iteration cap without producing a final "
                   "text response.]",
        iterations=max_iterations,
        tool_calls=tool_call_log,
        stopped_on="iteration_cap",
    )


def _strip_full_arrays(result: dict[str, Any]) -> dict[str, Any]:
    """Remove `_full` (the raw un-trimmed sweep) before sending to the LLM."""
    if "_full" in result:
        return {k: v for k, v in result.items() if k != "_full"}
    return result
