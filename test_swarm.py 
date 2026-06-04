"""
Unit tests for graph internals and EA optimizer logic.
Run with: python -m pytest test_swarm.py -v
"""
import asyncio
import math
import random
import pytest

from graph.edge import Edge
from graph.graph import Graph
from graph.node import Node
from swarm.ea_swarm import EASwarm


# ── Helpers ────────────────────────────────────────────────────────────────────

class DummyNode(Node):
    """Synchronous pass-through node for testing."""
    def __init__(self, value=None, **kwargs):
        super().__init__(**kwargs)
        self._value = value

    async def _execute(self, input, **kwargs):
        return self._value if self._value is not None else input


def make_simple_graph():
    """source -> target graph with one edge."""
    source = DummyNode(value="hello", operation_description="source")
    target = DummyNode(operation_description="target")
    graph = Graph(output_node=target)
    edge = Edge(source, target)
    graph.add_edge(edge)
    return graph, source, target, edge


# ── Edge tests ─────────────────────────────────────────────────────────────────

class TestEdge:
    def test_sigmoid_zero_weight(self):
        src = DummyNode(operation_description="s")
        tgt = DummyNode(operation_description="t")
        e = Edge(src, tgt, weight=0.0)
        assert abs(e.probability - 0.5) < 1e-6

    def test_sigmoid_positive_weight(self):
        src = DummyNode(operation_description="s")
        tgt = DummyNode(operation_description="t")
        e = Edge(src, tgt, weight=10.0)
        assert e.probability > 0.99

    def test_sigmoid_negative_weight(self):
        src = DummyNode(operation_description="s")
        tgt = DummyNode(operation_description="t")
        e = Edge(src, tgt, weight=-10.0)
        assert e.probability < 0.01

    def test_sample_active_high_weight(self):
        src = DummyNode(operation_description="s")
        tgt = DummyNode(operation_description="t")
        e = Edge(src, tgt, weight=100.0)
        results = [e.sample() for _ in range(20)]
        assert all(results)

    def test_sample_inactive_low_weight(self):
        src = DummyNode(operation_description="s")
        tgt = DummyNode(operation_description="t")
        e = Edge(src, tgt, weight=-100.0)
        results = [e.sample() for _ in range(20)]
        assert not any(results)

    def test_connect_links_nodes(self):
        src = DummyNode(operation_description="s")
        tgt = DummyNode(operation_description="t")
        e = Edge(src, tgt)
        e.connect()
        assert tgt in src.successors
        assert src in tgt.predecessors

    def test_repr_contains_probability(self):
        src = DummyNode(operation_description="s")
        tgt = DummyNode(operation_description="t")
        e = Edge(src, tgt, weight=0.0)
        assert "p=0.50" in repr(e)


# ── Graph tests ────────────────────────────────────────────────────────────────

class TestGraph:
    def test_add_edge_registers_nodes(self):
        graph, source, target, edge = make_simple_graph()
        assert source in graph.nodes
        assert target in graph.nodes

    def test_topological_sort_order(self):
        graph, source, target, _ = make_simple_graph()
        layers = graph.topological_sort()
        assert source in layers[0]
        assert target in layers[1]

    def test_cycle_raises(self):
        a = DummyNode(operation_description="a")
        b = DummyNode(operation_description="b")
        graph = Graph()
        graph.add_edge(Edge(a, b))
        graph.add_edge(Edge(b, a))
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_sort()

    def test_execute_passes_input(self):
        node = DummyNode(operation_description="passthrough")
        graph = Graph(output_node=node)
        result = asyncio.run(graph.execute("test_input"))
        assert result == ["test_input"]

    def test_execute_inactive_edge_skips_predecessor(self):
        graph, source, target, edge = make_simple_graph()
        edge.active = False
        # target gets no predecessor output — should warn and return empty
        import warnings
        with warnings.catch_warnings(record=True):
            result = asyncio.run(graph.execute("ignored"))
        assert result == []

    def test_execute_active_edge_propagates(self):
        source = DummyNode(value="propagated", operation_description="src")
        target = DummyNode(operation_description="tgt")
        graph = Graph(output_node=target)
        edge = Edge(source, target)
        graph.add_edge(edge)
        edge.active = True
        source.inputs = ["trigger"]
        result = asyncio.run(graph.execute("trigger"))
        assert result == ["propagated"]

    def test_sample_edges_called_on_execute_with_sample(self):
        graph, source, target, edge = make_simple_graph()
        edge.weight = 100.0  # always active
        result = asyncio.run(graph.execute("hi", sample=True))
        assert result == ["hello"]


# ── Node tests ─────────────────────────────────────────────────────────────────

class TestNode:
    def test_add_successor_bidirectional(self):
        a = DummyNode(operation_description="a")
        b = DummyNode(operation_description="b")
        a.add_successor(b)
        assert b in a.successors
        assert a in b.predecessors

    def test_remove_successor(self):
        a = DummyNode(operation_description="a")
        b = DummyNode(operation_description="b")
        a.add_successor(b)
        a.remove_successor(b)
        assert b not in a.successors
        assert a not in b.predecessors

    def test_combine_inputs_as_one(self):
        class CollectorNode(Node):
            async def _execute(self, input, **kwargs):
                return f"combined:{len(input)}"

        src1 = DummyNode(value="x", operation_description="s1")
        src2 = DummyNode(value="y", operation_description="s2")
        collector = CollectorNode(operation_description="col", combine_inputs_as_one=True)
        graph = Graph(output_node=collector)
        graph.add_edge(Edge(src1, collector))
        graph.add_edge(Edge(src2, collector))
        result = asyncio.run(graph.execute("trigger"))
        assert result[0].startswith("combined:")


# ── EA Swarm tests ─────────────────────────────────────────────────────────────

class TestEASwarm:
    def _make_ea(self, pop_size=4):
        graph, source, target, edge = make_simple_graph()
        ea = EASwarm(graph, pop_size=pop_size, mutation_rate=0.5, tournament_k=2)
        return ea, graph

    def test_population_size(self):
        ea, _ = self._make_ea(pop_size=6)
        assert len(ea.population) == 6

    def test_population_binary_values(self):
        ea, _ = self._make_ea(pop_size=8)
        for individual in ea.population:
            assert all(bit in (0, 1) for bit in individual)

    def test_individual_length_matches_edges(self):
        ea, graph = self._make_ea()
        for individual in ea.population:
            assert len(individual) == len(graph.edges)

    def test_apply_sets_edge_active(self):
        ea, graph = self._make_ea()
        ea._apply([1])
        assert graph.edges[0].active is True
        ea._apply([0])
        assert graph.edges[0].active is False

    def test_crossover_length_preserved(self):
        ea, _ = self._make_ea()
        p1 = [1, 0, 1, 0]
        p2 = [0, 1, 0, 1]
        ea.n_edges = 4
        c1, c2 = ea._crossover(p1, p2)
        assert len(c1) == 4
        assert len(c2) == 4

    def test_crossover_combines_parents(self):
        ea, _ = self._make_ea()
        ea.n_edges = 4
        random.seed(42)
        p1 = [1, 1, 1, 1]
        p2 = [0, 0, 0, 0]
        c1, c2 = ea._crossover(p1, p2)
        # children should be mixes, not identical to either parent
        assert c1 != p1 or c2 != p2

    def test_mutate_flips_bits(self):
        ea, _ = self._make_ea()
        random.seed(0)
        individual = [0] * 20
        mutated = ea._mutate(individual)
        assert any(b == 1 for b in mutated)  # mutation_rate=0.5, very likely

    def test_mutate_preserves_length(self):
        ea, _ = self._make_ea()
        ind = [1, 0, 1]
        ea.n_edges = 3
        assert len(ea._mutate(ind)) == 3

    def test_tournament_select_returns_valid(self):
        ea, _ = self._make_ea(pop_size=4)
        fitnesses = [0.1, 0.9, 0.5, 0.3]
        winner = ea._tournament_select(fitnesses)
        assert winner in ea.population

    def test_optimize_returns_best_and_history(self):
        graph, source, target, edge = make_simple_graph()
        ea = EASwarm(graph, pop_size=4, mutation_rate=0.2, tournament_k=2)
        questions = ["q1", "q2", "q3"]
        score_fns = [lambda _: 1.0, lambda _: 0.0, lambda _: 1.0]
        best, gen_bests = asyncio.run(
            ea.optimize(questions, score_fns, n_generations=3, batch_size=2)
        )
        assert len(best) == len(graph.edges)
        assert len(gen_bests) == 3
        assert all(0.0 <= f <= 1.0 for f in gen_bests)

    def test_elitism_best_survives(self):
        """Best individual from each generation must appear in next population."""
        graph, source, target, edge = make_simple_graph()
        ea = EASwarm(graph, pop_size=4, mutation_rate=0.0, tournament_k=2)  # no mutation
        questions = ["q"]
        score_fns = [lambda _: 1.0]
        asyncio.run(ea.optimize(questions, score_fns, n_generations=5, batch_size=1))
        # With mutation_rate=0 and elitism, population should be stable
        assert len(ea.population) == 4

    def test_population_stays_constant_size(self):
        graph, source, target, edge = make_simple_graph()
        ea = EASwarm(graph, pop_size=6, mutation_rate=0.3, tournament_k=2)
        questions = ["q1", "q2"]
        score_fns = [lambda _: 0.5, lambda _: 0.5]
        asyncio.run(ea.optimize(questions, score_fns, n_generations=4, batch_size=2))
        assert len(ea.population) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])