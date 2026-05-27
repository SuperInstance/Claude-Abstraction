"""PromptChain — sequential prompt pipelines with context passing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .prompt import PromptTemplate


# A step is either a PromptTemplate or a callable (context) -> str
Step = PromptTemplate | Callable[[dict[str, Any]], str]


@dataclass
class StepResult:
    """The outcome of a single chain step."""

    step_name: str
    output: str
    context: dict[str, Any]


@dataclass
class PromptChain:
    """A pipeline of prompt steps that pass context forward.

    Each step is a :class:`PromptTemplate` or a callable
    ``(context) -> str``.  The chain executes steps sequentially,
    feeding each step's output into the growing context.

    Example::

        chain = (
            PromptChain("analyze")
            .step(PromptTemplate("extract", "Extract key facts from:\n{{input}}"))
            .step(PromptTemplate("summarize", "Summarize these facts:\n{{extract_output}}"))
        )
        results = chain.run(input="Some long text...")
    """

    name: str
    steps: list[tuple[str, Step]] = field(default_factory=list)

    def step(self, s: Step, name: str | None = None) -> PromptChain:
        """Append a step and return *self* for chaining."""
        step_name = name or (s.name if isinstance(s, PromptTemplate) else f"step_{len(self.steps)}")
        self.steps.append((step_name, s))
        return self

    def run(self, initial_context: dict[str, Any] | None = None, **kwargs: Any) -> list[StepResult]:
        """Execute the chain and return per-step results."""
        ctx: dict[str, Any] = {**(initial_context or {}), **kwargs}
        results: list[StepResult] = []

        for step_name, s in self.steps:
            if isinstance(s, PromptTemplate):
                output = s.render(**ctx)
            else:
                output = s(ctx)

            results.append(StepResult(step_name=step_name, output=output, context=dict(ctx)))
            ctx[f"{step_name}_output"] = output

        return results

    def final_output(self, **kwargs: Any) -> str:
        """Run the chain and return only the last step's output."""
        results = self.run(**kwargs)
        return results[-1].output if results else ""

    def __repr__(self) -> str:
        return f"PromptChain(name={self.name!r}, steps={len(self.steps)})"
