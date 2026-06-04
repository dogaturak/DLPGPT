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
        in_degree: Dict[Node, int] = {node: len(node.predecessors) for node in self.nodes}
        queue: List[Node] = [n for n in self.nodes if in_degree[n] == 0]
        layers: List[List[Node]] = []

        while queue:
            layers.append(list(queue))
            next_queue = []
            for node in queue:
                for successor in node.successors:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        next_queue.append(successor)
            queue = next_queue

        if sum(len(l) for l in layers) != len(self.nodes):
            raise ValueError("Graph contains a cycle.")

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
        layers = self.topological_sort()

        for i, layer in enumerate(layers):
            if i == 0:
                for node in layer:
                    node.inputs = inputs if isinstance(inputs, list) else [inputs]
            if verbose:
                print(f"--- Layer {i}: {[repr(n) for n in layer]} ---")
                for node in layer:
                    print(f"  {repr(node)} inputs: {node.inputs}")
            await asyncio.gather(*[node.execute() for node in layer])
            if verbose:
                for node in layer:
                    print(f"  {repr(node)} outputs: {node.outputs}")

        if self.output_node is not None:
            return self.output_node.outputs
        return layers[-1][0].outputs if layers else []
