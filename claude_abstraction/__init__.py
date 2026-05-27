"""Claude Abstraction — prompt engineering patterns for managing LLM complexity."""

from .abstraction import AbstractionLayer
from .prompt import PromptTemplate
from .chain import PromptChain, StepResult
from .router import PromptRouter
from .optimizer import PromptOptimizer

__all__ = [
    "AbstractionLayer",
    "PromptTemplate",
    "PromptChain",
    "StepResult",
    "PromptRouter",
    "PromptOptimizer",
]
__version__ = "0.1.0"
