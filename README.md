# Claude Abstraction

Abstraction layers and prompt engineering patterns for managing LLM complexity.

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### AbstractionLayer — Nested Context

Compose layers of instructions, examples, and constraints:

```python
from claude_abstraction import AbstractionLayer

system = AbstractionLayer(
    "system",
    instructions="You are a helpful assistant.",
    constraints=["Be concise", "No hallucinations"],
)
persona = AbstractionLayer("persona", instructions="Speak like a pirate.")

merged = system + persona
print(merged.render())

# Use as context manager
with AbstractionLayer("session", instructions="Focus on Python."):
    # layer is active
    pass
```

### PromptTemplate — Variables & Conditionals

```python
from claude_abstraction import PromptTemplate

tpl = PromptTemplate(
    "review",
    "Review this {{language}} code:\n{{code}}{{#if strict}}\nBe extra strict.{{/if}}",
    defaults={"language": "Python"},
)

print(tpl.render(code="def foo(): pass", strict=True))
# "Review this Python code:\ndef foo(): pass\nBe extra strict."

# Compose templates
header = PromptTemplate("header", "You are {{role}}.\n\n")
body = PromptTemplate("body", "Task: {{task}}")
full = header + body
print(full.render(role="reviewer", task="Find bugs"))
```

### PromptChain — Sequential Pipelines

```python
from claude_abstraction import PromptChain, PromptTemplate

chain = (
    PromptChain("analyze")
    .step(PromptTemplate("extract", "Extract key facts from:\n{{input}}"))
    .step(PromptTemplate("summarize", "Summarize concisely:\n{{extract_output}}"))
)

results = chain.run(input="Long text about quantum computing...")
# Each step's output is available as {step_name}_output in subsequent steps

for r in results:
    print(f"[{r.step_name}] {r.output}")
```

### PromptRouter — Query Routing

```python
from claude_abstraction import PromptRouter, PromptTemplate

router = PromptRouter()
router.add("code", {"code", "function", "debug"}, PromptTemplate("code", "Code: {{input}}"))
router.add("chat", {"chat", "talk"}, PromptTemplate("chat", "Chat: {{input}}"))
router.set_default(PromptTemplate("general", "General: {{input}}"))

handler, route = router.route("Help me debug this function")
print(route)  # "code"

# Or render directly
print(router.render("Let's chat about weather", input="weather"))
```

### PromptOptimizer — Compression & Analysis

```python
from claude_abstraction import PromptOptimizer

opt = PromptOptimizer()

result = opt.optimize(
    "In order to write good code, please note that you should "
    "test test your your code code."
)
print(f"Saved {result.savings_pct}% ({result.original_chars} → {result.optimized_chars} chars)")
for suggestion in result.suggestions:
    print(f"  - {suggestion}")
```

## API Reference

| Class | Description |
|-------|-------------|
| `AbstractionLayer` | Nested, composable layers of prompt context |
| `PromptTemplate` | Reusable prompts with `{{var}}` and `{{#if var}}` syntax |
| `PromptChain` | Sequential prompt pipelines with context passing |
| `PromptRouter` | Keyword-based query routing to prompt handlers |
| `PromptOptimizer` | Token estimation, redundancy detection, compression |

## Design Principles

- **Zero dependencies** — only uses Python stdlib (+ pytest for tests)
- **Dataclasses throughout** — clean, typed, immutable-friendly
- **Composable by default** — every core type supports `+` operator
- **Framework agnostic** — no coupling to any specific LLM API

## Running Tests

```bash
python -m pytest tests/ -q
```

## License

MIT
