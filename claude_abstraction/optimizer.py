"""PromptOptimizer — measure token usage and suggest compressions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .prompt import PromptTemplate


@dataclass
class OptimizationResult:
    """Result of optimizing a prompt or template."""

    original_chars: int
    optimized_chars: int
    savings_pct: float
    original: str
    optimized: str
    suggestions: list[str]


@dataclass
class PromptOptimizer:
    """Analyze and compress prompts for efficiency.

    Provides token estimation, redundancy detection, and automatic
    compression suggestions.

    Example::

        opt = PromptOptimizer()
        result = opt.optimize("Please please respond respond with with no no duplicates duplicates.")
        print(result.suggestions)
    """

    chars_per_token: float = 4.0
    redundancy_threshold: int = 2  # max allowed repeats of same word

    def token_estimate(self, text: str) -> int:
        """Rough token count for *text*."""
        return max(1, int(len(text) / self.chars_per_token))

    def find_redundancies(self, text: str) -> list[str]:
        """Detect repeated consecutive words/phrases."""
        words = text.lower().split()
        redundancies: list[str] = []
        i = 0
        while i < len(words) - 1:
            run = 1
            while i + run < len(words) and words[i + run] == words[i]:
                run += 1
            if run >= self.redundancy_threshold:
                redundancies.append(f"'{words[i]}' repeated {run} times")
            i += run
        return redundancies

    def find_long_sentences(self, text: str, max_words: int = 40) -> list[str]:
        """Flag sentences exceeding *max_words*."""
        sentences = re.split(r'[.!?]+', text)
        long: list[str] = []
        for s in sentences:
            s = s.strip()
            if s and len(s.split()) > max_words:
                long.append(s[:80] + "...")
        return long

    def suggest_compressions(self, text: str) -> list[str]:
        """Generate compression suggestions."""
        suggestions: list[str] = []

        # Redundancy
        redundancies = self.find_redundancies(text)
        for r in redundancies:
            suggestions.append(f"Remove redundant repetition: {r}")

        # Long sentences
        long = self.find_long_sentences(text)
        for s in long:
            suggestions.append(f"Break up long sentence: {s}")

        # Filler phrases
        filler_patterns = [
            (r"(?i)\bplease\s+note\s+that\b", "Remove filler 'please note that'"),
            (r"(?i)\bit\s+is\s+important\s+to\s+(note|remember)\b", "Remove filler 'it is important to note/remember'"),
            (r"(?i)\bin\s+order\s+to\b", "Replace 'in order to' with 'to'"),
            (r"(?i)\bat\s+this\s+point\s+in\s+time\b", "Replace 'at this point in time' with 'now'"),
            (r"(?i)\bdue\s+to\s+the\s+fact\s+that\b", "Replace 'due to the fact that' with 'because'"),
            (r"(?i)\bfor\s+all\s+intents\s+and\s+purposes\b", "Remove 'for all intents and purposes'"),
        ]
        for pattern, msg in filler_patterns:
            if re.search(pattern, text):
                suggestions.append(msg)

        # Trailing whitespace
        if text != text.rstrip():
            suggestions.append("Remove trailing whitespace")

        # Multiple blank lines
        if "\n\n\n" in text:
            suggestions.append("Collapse multiple blank lines to double")

        return suggestions

    def compress(self, text: str) -> str:
        """Apply automatic compressions to *text*."""
        result = text

        # Remove consecutive duplicate words
        result = re.sub(r'\b(\w+)\s+\1\b', r'\1', result, flags=re.IGNORECASE)

        # Collapse multiple blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)

        # Strip trailing whitespace per line
        result = '\n'.join(line.rstrip() for line in result.split('\n'))

        # Simple filler replacements
        replacements = [
            (r"(?i)\bin order to\b", "to"),
            (r"(?i)\bdue to the fact that\b", "because"),
            (r"(?i)\bat this point in time\b", "now"),
            (r"(?i)\bfor all intents and purposes\b", "essentially"),
        ]
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)

        return result.strip()

    def optimize(self, text: str) -> OptimizationResult:
        """Full optimization pass: compress and report."""
        suggestions = self.suggest_compressions(text)
        optimized = self.compress(text)
        original_chars = len(text)
        optimized_chars = len(optimized)
        savings = ((original_chars - optimized_chars) / original_chars * 100) if original_chars else 0

        return OptimizationResult(
            original_chars=original_chars,
            optimized_chars=optimized_chars,
            savings_pct=round(savings, 1),
            original=text,
            optimized=optimized,
            suggestions=suggestions,
        )

    def optimize_template(self, template: PromptTemplate, **kwargs: Any) -> OptimizationResult:
        """Optimize a rendered template."""
        rendered = template.render(**kwargs)
        return self.optimize(rendered)

    def __repr__(self) -> str:
        return f"PromptOptimizer(chars_per_token={self.chars_per_token})"
