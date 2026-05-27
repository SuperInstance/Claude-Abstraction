"""Tests for claude_abstraction package."""

import pytest

from claude_abstraction import AbstractionLayer, PromptTemplate, PromptChain, PromptRouter, PromptOptimizer


# ---------------------------------------------------------------------------
# AbstractionLayer
# ---------------------------------------------------------------------------
class TestAbstractionLayer:
    def test_basic_render(self):
        layer = AbstractionLayer("sys", instructions="You are helpful.")
        assert "You are helpful." in layer.render()

    def test_examples_render(self):
        layer = AbstractionLayer("ex", examples=["Q: hi\nA: hello"])
        assert "Example:" in layer.render()
        assert "Q: hi" in layer.render()

    def test_constraints_render(self):
        layer = AbstractionLayer("rules", constraints=["No profanity", "Be concise"])
        rendered = layer.render()
        assert "No profanity" in rendered
        assert "Be concise" in rendered

    def test_merge(self):
        a = AbstractionLayer("a", instructions="A", constraints=["c1"])
        b = AbstractionLayer("b", instructions="B", constraints=["c2"])
        merged = a + b
        assert "A" in merged.render()
        assert "B" in merged.render()
        assert merged.name == "a+b"

    def test_add_child(self):
        parent = AbstractionLayer("parent", instructions="Top")
        child = AbstractionLayer("child", instructions="Nested")
        family = parent.add_child(child)
        assert len(family.children) == 1
        rendered = family.render()
        assert "Top" in rendered
        assert "Nested" in rendered

    def test_flatten(self):
        root = AbstractionLayer("root")
        child = AbstractionLayer("c1")
        grandchild = AbstractionLayer("gc")
        tree = root.add_child(child.add_child(grandchild))
        flat = tree.flatten()
        assert len(flat) == 3

    def test_priority_sorting(self):
        low = AbstractionLayer("low", instructions="Low", priority=0)
        high = AbstractionLayer("high", instructions="High", priority=10)
        merged = low + high
        rendered = merged.render()
        # Both instructions should be present; high priority content renders first
        assert "Low" in rendered
        assert "High" in rendered
        # Verify the merged priority is the max
        assert merged.priority == 10

    def test_context_manager(self):
        layer = AbstractionLayer("test")
        assert not layer.active
        with layer:
            assert layer.active
        assert not layer.active

    def test_token_estimate(self):
        layer = AbstractionLayer("t", instructions="x" * 40)
        est = layer.token_estimate()
        # Render adds a [t] header, so total chars > 40
        assert est >= 10

    def test_repr(self):
        layer = AbstractionLayer("demo", priority=5)
        r = repr(layer)
        assert "demo" in r
        assert "priority=5" in r


# ---------------------------------------------------------------------------
# PromptTemplate
# ---------------------------------------------------------------------------
class TestPromptTemplate:
    def test_basic_render(self):
        tpl = PromptTemplate("greet", "Hello {{name}}!")
        assert tpl.render(name="World") == "Hello World!"

    def test_defaults(self):
        tpl = PromptTemplate("greet", "Hello {{name}}!", defaults={"name": "friend"})
        assert tpl.render() == "Hello friend!"
        assert tpl.render(name="Alice") == "Hello Alice!"

    def test_conditional_true(self):
        tpl = PromptTemplate("t", "Hi{{#if formal}} and welcome{{/if}}")
        assert tpl.render(formal=True) == "Hi and welcome"

    def test_conditional_false(self):
        tpl = PromptTemplate("t", "Hi{{#if formal}} and welcome{{/if}}")
        assert tpl.render(formal=False) == "Hi"

    def test_unresolved_vars(self):
        tpl = PromptTemplate("t", "Hello {{name}}!")
        assert tpl.render() == "Hello {{name}}!"

    def test_variables(self):
        tpl = PromptTemplate("t", "{{a}} and {{b}} and {{a}}")
        assert tpl.variables() == {"a", "b"}

    def test_validate(self):
        tpl = PromptTemplate("t", "{{a}} {{b}}", defaults={"b": "x"})
        missing = tpl.validate()
        assert missing == ["a"]

    def test_validate_with_kwargs(self):
        tpl = PromptTemplate("t", "{{a}} {{b}}")
        missing = tpl.validate(a=1)
        assert missing == ["b"]

    def test_compose(self):
        t1 = PromptTemplate("a", "Part A {{x}}")
        t2 = PromptTemplate("b", "Part B {{y}}")
        combined = t1 + t2
        rendered = combined.render(x=1, y=2)
        assert "Part A 1" in rendered
        assert "Part B 2" in rendered

    def test_repr(self):
        tpl = PromptTemplate("demo", "{{x}}")
        assert "demo" in repr(tpl)


# ---------------------------------------------------------------------------
# PromptChain
# ---------------------------------------------------------------------------
class TestPromptChain:
    def test_single_step(self):
        chain = PromptChain("c").step(PromptTemplate("s1", "Result: {{input}}"))
        results = chain.run(input="hello")
        assert len(results) == 1
        assert results[0].output == "Result: hello"

    def test_multi_step_context(self):
        chain = (
            PromptChain("c")
            .step(PromptTemplate("extract", "Facts from: {{input}}"))
            .step(PromptTemplate("summarize", "Summary of: {{extract_output}}"))
        )
        results = chain.run(input="data")
        assert len(results) == 2
        assert "Facts from: data" == results[0].output
        assert "Summary of: Facts from: data" == results[1].output

    def test_callable_step(self):
        chain = PromptChain("c").step(lambda ctx: f"Got: {ctx['x']}")
        results = chain.run(x=42)
        assert results[0].output == "Got: 42"

    def test_final_output(self):
        chain = (
            PromptChain("c")
            .step(PromptTemplate("a", "first"))
            .step(PromptTemplate("b", "second"))
        )
        assert chain.final_output() == "second"

    def test_empty_chain(self):
        chain = PromptChain("empty")
        assert chain.final_output() == ""

    def test_repr(self):
        chain = PromptChain("c").step(PromptTemplate("s", "x"))
        assert "steps=1" in repr(chain)


# ---------------------------------------------------------------------------
# PromptRouter
# ---------------------------------------------------------------------------
class TestPromptRouter:
    def test_route_by_keyword(self):
        router = PromptRouter()
        router.add("code", {"code", "function"}, PromptTemplate("code", "Code: {{query}}"))
        router.add("chat", {"chat", "talk"}, PromptTemplate("chat", "Chat: {{query}}"))
        handler, name = router.route("debug this function")
        assert name == "code"

    def test_priority_tiebreak(self):
        router = PromptRouter()
        router.add("low", {"help"}, PromptTemplate("l", "Low"), priority=1)
        router.add("high", {"help"}, PromptTemplate("h", "High"), priority=10)
        _, name = router.route("I need help")
        assert name == "high"

    def test_default_fallback(self):
        router = PromptRouter()
        router.set_default(PromptTemplate("def", "Default: {{query}}"))
        handler, name = router.route("something random")
        assert name == "__default__"

    def test_no_match_raises(self):
        router = PromptRouter()
        with pytest.raises(ValueError, match="No matching route"):
            router.route("anything")

    def test_render(self):
        router = PromptRouter()
        router.add("code", {"code"}, PromptTemplate("c", "Code mode: {{input}}"))
        result = router.render("write code", input="hello world")
        assert result == "Code mode: hello world"

    def test_callable_handler(self):
        router = PromptRouter()
        router.add("fn", {"test"}, lambda ctx: f"Handled: {ctx['query']}")
        handler, name = router.route("run test")
        output = handler({"query": "run test"})
        assert output == "Handled: run test"

    def test_repr(self):
        router = PromptRouter()
        router.add("a", {"x"}, PromptTemplate("t", "x"))
        assert "routes=1" in repr(router)


# ---------------------------------------------------------------------------
# PromptOptimizer
# ---------------------------------------------------------------------------
class TestPromptOptimizer:
    def test_token_estimate(self):
        opt = PromptOptimizer()
        assert opt.token_estimate("a" * 40) == 10

    def test_find_redundancies(self):
        opt = PromptOptimizer()
        reds = opt.find_redundancies("the the the cat")
        assert len(reds) == 1
        assert "the" in reds[0]

    def test_no_redundancies(self):
        opt = PromptOptimizer()
        assert opt.find_redundancies("the cat sat") == []

    def test_long_sentences(self):
        opt = PromptOptimizer()
        long = " ".join(["word"] * 50) + "."
        result = opt.find_long_sentences(long)
        assert len(result) == 1

    def test_filler_suggestions(self):
        opt = PromptOptimizer()
        suggestions = opt.suggest_compressions("In order to succeed, please note that you must try.")
        msgs = " ".join(suggestions)
        assert "in order to" in msgs.lower()
        assert "please note that" in msgs.lower()

    def test_compress_duplicate_words(self):
        opt = PromptOptimizer()
        compressed = opt.compress("hello hello world world")
        assert compressed == "hello world"

    def test_compress_filler(self):
        opt = PromptOptimizer()
        compressed = opt.compress("In order to run, due to the fact that it rains")
        assert "In order to" not in compressed
        assert "because" in compressed
        assert "to run" in compressed

    def test_compress_blank_lines(self):
        opt = PromptOptimizer()
        compressed = opt.compress("line1\n\n\n\nline2")
        assert "\n\n\n" not in compressed

    def test_optimize(self):
        opt = PromptOptimizer()
        result = opt.optimize("hello hello world world")
        assert result.savings_pct > 0
        assert result.optimized == "hello world"
        assert result.original == "hello hello world world"

    def test_optimize_template(self):
        opt = PromptOptimizer()
        tpl = PromptTemplate("t", "{{x}} {{x}}")
        result = opt.optimize_template(tpl, x="hello")
        assert "hello" in result.optimized

    def test_zero_length(self):
        opt = PromptOptimizer()
        result = opt.optimize("")
        assert result.savings_pct == 0
        assert result.original_chars == 0

    def test_repr(self):
        opt = PromptOptimizer(chars_per_token=3.0)
        assert "3.0" in repr(opt)
