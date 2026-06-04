import asyncio
import uuid
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.edge import Edge


class Node(ABC):
    def __init__(
        self,
        operation_description: str = "",
        node_id: Optional[str] = None,
        combine_inputs_as_one: bool = False,
    ):
        self.id = node_id or str(uuid.uuid4())[:8]
        self.operation_description = operation_description
        self.combine_inputs_as_one = combine_inputs_as_one
        self.predecessors: List["Node"] = []
        self.successors: List["Node"] = []
        self.incoming_edges: Dict["Node", "Edge"] = {}
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []

    @property
    def node_name(self) -> str:
        return self.__class__.__name__

    def add_predecessor(self, node: "Node") -> None:
        if node not in self.predecessors:
            self.predecessors.append(node)
            if self not in node.successors:
                node.successors.append(self)

    def add_successor(self, node: "Node") -> None:
        if node not in self.successors:
            self.successors.append(node)
            if self not in node.predecessors:
                node.predecessors.append(self)

    def remove_predecessor(self, node: "Node") -> None:
        if node in self.predecessors:
            self.predecessors.remove(node)
            if self in node.successors:
                node.successors.remove(self)

    def remove_successor(self, node: "Node") -> None:
        if node in self.successors:
            self.successors.remove(node)
            if self in node.predecessors:
                node.predecessors.remove(self)

    def _active_predecessors(self) -> List["Node"]:
        result = []
        for pred in self.predecessors:
            edge = self.incoming_edges.get(pred)
            if edge is None or edge.active:
                result.append(pred)
        return result

    async def execute(self, **kwargs) -> None:
        self.outputs = []
        tasks = []
        active_preds = self._active_predecessors()

        if not self.inputs and active_preds:
            if self.combine_inputs_as_one:
                combined = []
                for pred in active_preds:
                    if isinstance(pred.outputs, list):
                        combined.extend(pred.outputs)
                tasks.append(asyncio.create_task(self._execute(combined, **kwargs)))
            else:
                for pred in active_preds:
                    for output in pred.outputs or []:
                        tasks.append(asyncio.create_task(self._execute(output, **kwargs)))
        elif self.inputs:
            tasks = [asyncio.create_task(self._execute(inp, **kwargs)) for inp in self.inputs]
        else:
            warnings.warn(f"Node {self.node_name} received no input.")
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                warnings.warn(f"Node {self.node_name} failed: {result}")
            else:
                if not isinstance(result, list):
                    result = [result]
                self.outputs.extend(result)

    def __repr__(self) -> str:
        label = self.operation_description or self.id
        return f"{self.node_name}({label})"

    @abstractmethod
    async def _execute(self, input: Any, **kwargs) -> Any:
        """Process a single input and return the result."""
