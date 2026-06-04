"""
Integration tests for graph execution with real Ollama (llama3.1:8b).
Run with: python -m pytest test_graph.py -v
"""
import asyncio
import pytest

from graph import CombineAnswerNode, Edge, Graph, LLMNode


MODEL = "llama3.1"

MCQA_PROMPT = (
    "You are a knowledgeable expert in question answering. I will ask you a question. "
    "I will also give you 4 answers enumerated as A, B, C and D. Only one answer out of "
    "the offered 4 is correct. You must choose the correct answer to the question. Your "
    "response must be one of the 4 letters: A, B, C or D, corresponding to the correct "
    "answer. Only one letter (A, B, C or D) is allowed in your answer."
)

SAMPLE_QUESTION = (
    "Question: What is the chemical symbol for water?\n"
    "A) CO2\nB) H2O\nC) NaCl\nD) O2"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.run(coro)


def extract_letter(outputs):
    import re
    if not outputs:
        return ""
    match = re.search(r"\b([A-D])\b", str(outputs[0]).upper())
    return match.group(1) if match else ""


# ── Single LLMNode ─────────────────────────────────────────────────────────────

class TestSingleLLMNode:
    def test_returns_output(self):
        node = LLMNode(MCQA_PROMPT, model=MODEL, operation_description="IO")
        graph = Graph(output_node=node)
        result = run(graph.execute(SAMPLE_QUESTION))
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert len(result[0]) > 0

    def test_returns_valid_letter(self):
        node = LLMNode(MCQA_PROMPT, model=MODEL, operation_description="IO")
        graph = Graph(output_node=node)
        result = run(graph.execute(SAMPLE_QUESTION))
        letter = extract_letter(result)
        assert letter in ("A", "B", "C", "D")

    def test_correct_answer_water(self):
        node = LLMNode(MCQA_PROMPT, model=MODEL, operation_description="IO")
        graph = Graph(output_node=node)
        result = run(graph.execute(SAMPLE_QUESTION))
        assert extract_letter(result) == "B"


# ── Two-node pipeline ──────────────────────────────────────────────────────────

class TestTwoNodeGraph:
    def test_pipeline_executes(self):
        solver = LLMNode(MCQA_PROMPT, model=MODEL, operation_description="Solver")
        aggregator = CombineAnswerNode(
            system_prompt=(
                "You are given an answer to a multiple-choice question as a single letter "
                "(A, B, C, or D). Output only that letter."
            ),
            model=MODEL,
            operation_description="Aggregator",
        )
        graph = Graph(output_node=aggregator)
        graph.add_edge(Edge(solver, aggregator))
        result = run(graph.execute(SAMPLE_QUESTION))
        assert isinstance(result, list)
        assert len(result) > 0

    def test_pipeline_returns_letter(self):
        solver = LLMNode(MCQA_PROMPT, model=MODEL, operation_description="Solver")
        aggregator = CombineAnswerNode(
            system_prompt=(
                "Two agents answered a multiple-choice question, each with a single letter "
                "(A, B, C, or D). Choose the most common. Output only one letter."
            ),
            model=MODEL,
            operation_description="Aggregator",
        )
        graph = Graph(output_node=aggregator)
        graph.add_edge(Edge(solver, aggregator))
        result = run(graph.execute(SAMPLE_QUESTION))
        letter = extract_letter(result)
        assert letter in ("A", "B", "C", "D")


# ── Full swarm graph (direct + adversarial + aggregator) ──────────────────────

class TestSwarmGraph:
    def _build(self):
        direct = LLMNode(MCQA_PROMPT, model=MODEL, operation_description="direct_answer")
        adversarial = LLMNode(
            "Answer a lie to the following question:",
            model=MODEL,
            operation_description="adversarial",
        )
        aggregator = CombineAnswerNode(
            system_prompt=(
                "Two agents have answered a multiple-choice question, each giving a single "
                "letter (A, B, C, or D). Choose the letter that appears most often. If tied, "
                "use your best judgement. Output only a single letter: A, B, C, or D."
            ),
            model=MODEL,
            operation_description="Aggregator",
        )
        graph = Graph(output_node=aggregator)
        for agent in (direct, adversarial):
            graph.add_edge(Edge(agent, aggregator))
        return graph

    def test_swarm_executes(self):
        graph = self._build()
        for e in graph.edges:
            e.active = True
        result = run(graph.execute(SAMPLE_QUESTION, sample=False))
        assert isinstance(result, list)
        assert len(result) > 0

    def test_swarm_returns_letter(self):
        graph = self._build()
        for e in graph.edges:
            e.active = True
        result = run(graph.execute(SAMPLE_QUESTION, sample=False))
        letter = extract_letter(result)
        assert letter in ("A", "B", "C", "D")

    def test_swarm_with_sampling(self):
        graph = self._build()
        for e in graph.edges:
            e.weight = 100.0  # force active
        result = run(graph.execute(SAMPLE_QUESTION, sample=True))
        assert isinstance(result, list)

    def test_inactive_edges_still_returns(self):
        """With all edges inactive, output node gets no predecessor input — should handle gracefully."""
        graph = self._build()
        for e in graph.edges:
            e.active = False
        import warnings
        with warnings.catch_warnings(record=True):
            result = run(graph.execute(SAMPLE_QUESTION, sample=False))
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])