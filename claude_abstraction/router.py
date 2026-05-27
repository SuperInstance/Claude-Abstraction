"""PromptRouter — direct queries to the best-matching prompt strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .prompt import PromptTemplate

# A route handler: either a PromptTemplate or a callable
Handler = PromptTemplate | Callable[[dict[str, Any]], str]


@dataclass
class Route:
    """A single route entry."""

    name: str
    keywords: set[str]
    handler: Handler
    priority: int = 0


@dataclass
class PromptRouter:
    """Route user queries to the most appropriate prompt handler.

    Routes are matched by keyword overlap.  The route with the highest
    ``priority`` among those that match wins.

    Example::

        router = PromptRouter()
        router.add("code", {"code", "function", "debug"}, code_template)
        router.add("chat", {"chat", "talk", "conversation"}, chat_template)

        handler, route = router.route("Help me debug this function")
        print(handler.render(input="Help me debug this function"))
    """

    routes: list[Route] = field(default_factory=list)
    default_handler: Handler | None = None

    def add(
        self,
        name: str,
        keywords: set[str] | list[str],
        handler: Handler,
        priority: int = 0,
    ) -> PromptRouter:
        """Register a route. Returns *self* for chaining."""
        kw = set(keywords) if isinstance(keywords, list) else keywords
        self.routes.append(Route(name=name, keywords=kw, handler=handler, priority=priority))
        return self

    def set_default(self, handler: Handler) -> PromptRouter:
        """Set a fallback handler for unmatched queries."""
        self.default_handler = handler
        return self

    def route(self, query: str) -> tuple[Handler, str]:
        """Find the best handler for *query*.

        Returns ``(handler, route_name)``.  Falls back to
        ``default_handler`` if nothing matches.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        best: Route | None = None
        best_overlap = 0

        for r in self.routes:
            overlap = len(query_words & {kw.lower() for kw in r.keywords})
            if overlap > best_overlap or (overlap == best_overlap and best and r.priority > best.priority):
                best = r
                best_overlap = overlap

        if best and best_overlap > 0:
            return best.handler, best.name

        if self.default_handler is not None:
            return self.default_handler, "__default__"

        raise ValueError(f"No matching route for query: {query!r}")

    def render(self, query: str, **kwargs: Any) -> str:
        """Route the query and render the handler with the given kwargs."""
        handler, _ = self.route(query)
        if isinstance(handler, PromptTemplate):
            return handler.render(**kwargs)
        return handler(kwargs)

    def __repr__(self) -> str:
        return f"PromptRouter(routes={len(self.routes)})"
