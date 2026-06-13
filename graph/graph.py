import asyncio
from typing import Any, Dict, List
from graph.node import Node
from graph.edge import Edge


class Graph:
    def __init__(
        self,
        nodes: List[Node] = None,
        edges: List[Edge] = None,
        output_node: Node = None,
    ):
        self.nodes: List[Node] = nodes or []
        self.edges: List[Edge] = edges or []
        self.output_node: Node = output_node
        if output_node is not None and output_node not in self.nodes:
            self.nodes.append(output_node)

    def add_node(self, node: Node) -> None:
        if node not in self.nodes:
            self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        if edge not in self.edges:
            self.edges.append(edge)
            edge.connect()
            if edge.source not in self.nodes:
                self.nodes.append(edge.source)
            if edge.target not in self.nodes:
                self.nodes.append(edge.target)

    def topological_sort(self) -> List[List[Node]]:
        active_predecessors: Dict[Node, set] = {
            node: set() for node in self.nodes}
        active_successors: Dict[Node, List[Node]] = {
            node: [] for node in self.nodes}

        for edge in self.edges:
            if edge.active:
                active_predecessors[edge.target].add(edge.source)
                active_successors[edge.source].append(edge.target)

        # Only execute nodes that contribute to the output (backward reachability).
        # Nodes whose outgoing edges are all inactive are excluded so they don't
        # make unnecessary LLM calls.
        if self.output_node is not None:
            contributing: set = set()
            stack = [self.output_node]
            while stack:
                node = stack.pop()
                if node not in contributing:
                    contributing.add(node)
                    for pred in active_predecessors[node]:
                        stack.append(pred)
        else:
            contributing = set(self.nodes)

        in_degree = {
            node: len(active_predecessors[node]) for node in contributing}
        queue = [n for n in self.nodes if n in contributing and in_degree[n] == 0]
        layers = []

        while queue:
            layers.append(list(queue))
            next_queue = []
            for node in queue:
                for successor in active_successors[node]:
                    if successor in contributing:
                        in_degree[successor] -= 1
                        if in_degree[successor] == 0:
                            next_queue.append(successor)
            queue = next_queue

        reachable = {node for layer in layers for node in layer}
        if self.output_node not in reachable:
            raise ValueError(
                "Output node is unreachable with current edge configuration.")

        return layers

    def describe(self) -> None:
        layers = self.topological_sort()
        print("=== Graph Structure ===")
        for i, layer in enumerate(layers):
            print(f"Layer {i}: {[repr(n) for n in layer]}")
        print("\nEdges:")
        for edge in self.edges:
            print(f"  {repr(edge)}")
        if self.output_node:
            print(f"\nOutput node: {repr(self.output_node)}")
        print("======================\n")

    def sample_edges(self) -> None:
        """Sample all edge activities from their current weights."""
        for edge in self.edges:
            edge.sample()

    async def execute(self, inputs: Any, verbose: bool = False, sample: bool = False) -> List[Any]:
        if sample:
            self.sample_edges()

        # Reset all node state before each execution
        for node in self.nodes:
            node.inputs = []
            node.outputs = []

        layers = self.topological_sort()

        for i, layer in enumerate(layers):
            if i == 0:
                for node in layer:
                    node.inputs = inputs if isinstance(
                        inputs, list) else [inputs]
            if verbose:
                print(f"--- Layer {i}: {[repr(n) for n in layer]} ---")
                for node in layer:
                    print(f"  {repr(node)} inputs: {node.inputs}")
            await asyncio.gather(*[node.execute() for node in layer])
            if verbose:
                for node in layer:
                    print(f"  {repr(node)} outputs: {node.outputs}")

        if self.output_node is not None:
            executed = [n for layer in layers for n in layer if n is not self.output_node]
            failed = [n for n in executed if not n.outputs]
            if executed and len(failed) == len(executed):
                raise RuntimeError(
                    f"All {len(executed)} agents produced no output"
                    " — Ollama is likely down or unresponsive."
                )
            elif executed and len(failed) > 0:
                print(
                    f"WARNING: {len(failed)}/{len(executed)} agents produced no output.",
                    flush=True,
                )
            return self.output_node.outputs
        return layers[-1][0].outputs if layers else []
