"""AbstractionLayer — nested, composable layers of prompt context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class AbstractionLayer:
    """A named, nestable layer of prompt context.

    Layers can be composed (merged), nested (parent/child), and rendered
    into a single prompt string. Use ``context()`` as a context manager
    to temporarily activate a layer.

    Example::

        system = AbstractionLayer("system", instructions="You are a helpful assistant.")
        persona = AbstractionLayer("persona", instructions="Speak like a pirate.")
        merged = system + persona
        print(merged.render())
    """

    name: str
    instructions: str = ""
    examples: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    children: list[AbstractionLayer] = field(default_factory=list)
    priority: int = 0
    _active: bool = field(default=False, repr=False)

    # ---- composition ----

    def __add__(self, other: AbstractionLayer) -> AbstractionLayer:
        """Merge two layers into a new composite layer."""
        return self.merge(other)

    def merge(self, other: AbstractionLayer, name: str | None = None) -> AbstractionLayer:
        """Create a new layer combining *self* and *other*.

        Instructions are concatenated; examples, constraints, and children
        are combined; priority takes the higher value.
        """
        return AbstractionLayer(
            name=name or f"{self.name}+{other.name}",
            instructions=f"{self.instructions}\n{other.instructions}".strip(),
            examples=[*self.examples, *other.examples],
            constraints=[*self.constraints, *other.constraints],
            children=[*self.children, *other.children],
            priority=max(self.priority, other.priority),
        )

    # ---- nesting ----

    def add_child(self, child: AbstractionLayer) -> AbstractionLayer:
        """Return a copy of this layer with *child* appended."""
        new = AbstractionLayer(
            name=self.name,
            instructions=self.instructions,
            examples=list(self.examples),
            constraints=list(self.constraints),
            children=[*self.children, child],
            priority=self.priority,
        )
        return new

    def flatten(self) -> list[AbstractionLayer]:
        """Return this layer and all descendants in depth-first order."""
        result: list[AbstractionLayer] = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result

    # ---- rendering ----

    def render(self, separator: str = "\n\n") -> str:
        """Render the full layer tree into a single prompt string."""
        parts: list[str] = []
        for layer in sorted(self.flatten(), key=lambda l: l.priority, reverse=True):
            sections: list[str] = []
            if layer.instructions:
                sections.append(layer.instructions)
            for ex in layer.examples:
                sections.append(f"Example:\n{ex}")
            if layer.constraints:
                block = "\n".join(f"- {c}" for c in layer.constraints)
                sections.append(f"Constraints:\n{block}")
            if sections:
                header = f"[{layer.name}]" if layer.name else ""
                parts.append(f"{header}\n{separator.join(sections)}" if header else separator.join(sections))
        return separator.join(parts)

    # ---- context manager ----

    def __enter__(self) -> AbstractionLayer:
        self._active = True
        return self

    def __exit__(self, *exc: object) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    # ---- helpers ----

    def token_estimate(self, chars_per_token: float = 4.0) -> int:
        """Rough token count estimate for the rendered output."""
        return int(len(self.render()) / chars_per_token)

    def __repr__(self) -> str:
        return (
            f"AbstractionLayer(name={self.name!r}, priority={self.priority}, "
            f"children={len(self.children)}, active={self._active})"
        )
