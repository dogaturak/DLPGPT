import math
import random
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graph.node import Node


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Edge:
    def __init__(
        self,
        source: "Node",
        target: "Node",
        transform: Optional[Callable[[Any], Any]] = None,
        weight: float = 0.0,
        temperature: float = 1.0,
    ):
        self.source = source
        self.target = target
        self.transform = transform or (lambda x: x)
        self.weight = weight  # logit; sigmoid(0) = 0.5
        self.temperature = temperature  # <1 sharper, >1 flatter
        self.active: bool = True  # set by sample() during optimization

    @property
    def probability(self) -> float:
        return _sigmoid(self.weight / self.temperature)

    def sample(self) -> bool:
        """Sample edge activity from Bernoulli(sigmoid(weight))."""
        self.active = random.random() < self.probability
        return self.active

    def connect(self) -> None:
        self.source.add_successor(self.target)
        self.target.incoming_edges[self.source] = self

    def disconnect(self) -> None:
        self.source.remove_successor(self.target)
        self.target.incoming_edges.pop(self.source, None)

    def apply(self, data: Any) -> Any:
        return self.transform(data)

    def __repr__(self) -> str:
        return f"Edge({self.source.id} -> {self.target.id}, p={self.probability:.2f})"
