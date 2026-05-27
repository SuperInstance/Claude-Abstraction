"""PromptTemplate — reusable prompts with variables, conditionals, and composition."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_VAR_RE = re.compile(r"\{\{(\w+)\}\}")
_COND_RE = re.compile(r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}", re.DOTALL)


@dataclass
class PromptTemplate:
    """A reusable prompt template with variable interpolation and conditionals.

    Variables use ``{{name}}`` syntax.  Conditionals use
    ``{{#if name}}...{{/if}}`` blocks that render only when the variable
    is truthy.

    Example::

        tpl = PromptTemplate(
            "greet",
            "Hello {{name}}!{{#if excited}} You're awesome!{{/if}}",
        )
        print(tpl.render(name="Alice", excited=True))
    """

    name: str
    template: str
    defaults: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def render(self, **kwargs: Any) -> str:
        """Render the template with the given variables."""
        merged = {**self.defaults, **kwargs}

        # Handle conditionals first
        def _cond_replace(match: re.Match[str]) -> str:
            var = match.group(1)
            body = match.group(2)
            return body if merged.get(var) else ""

        result = _COND_RE.sub(_cond_replace, self.template)

        # Then interpolate variables
        def _var_replace(match: re.Match[str]) -> str:
            var = match.group(1)
            if var not in merged:
                return match.group(0)  # leave unresolved
            return str(merged[var])

        result = _VAR_RE.sub(_var_replace, result)
        return result.strip()

    def variables(self) -> set[str]:
        """Return all variable names referenced in the template."""
        return set(_VAR_RE.findall(self.template))

    def validate(self, **kwargs: Any) -> list[str]:
        """Check for missing required variables (those without defaults)."""
        required = self.variables() - set(self.defaults.keys())
        missing = required - set(kwargs.keys())
        return sorted(missing)

    def compose(self, other: PromptTemplate, separator: str = "\n\n") -> PromptTemplate:
        """Create a new template by concatenating *self* and *other*."""
        return PromptTemplate(
            name=f"{self.name}+{other.name}",
            template=f"{self.template}{separator}{other.template}",
            defaults={**self.defaults, **other.defaults},
            description=f"Composition of {self.name} and {other.name}",
        )

    def __add__(self, other: PromptTemplate) -> PromptTemplate:
        return self.compose(other)

    def __repr__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, vars={self.variables()})"
