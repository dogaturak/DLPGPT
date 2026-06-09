import math
from typing import Any, Callable, Dict, List, Tuple, Union
from graph.edge import Edge
from graph.graph import Graph

QuestionList = Union[List[Any], List[Tuple[Any, Callable]]]


class Swarm:
    def __init__(
        self,
        graph: Graph,
        lr: float = 0.01,
        baseline_decay: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.graph = graph
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.baseline = 0.0
        self._t = 0
        self._m: Dict[int, float] = {id(e): 0.0 for e in graph.edges}
        self._v: Dict[int, float] = {id(e): 0.0 for e in graph.edges}

    def _register_edge(self, edge: Edge) -> None:
        key = id(edge)
        if key not in self._m:
            self._m[key] = 0.0
            self._v[key] = 0.0

    async def step(self, question: Any, score_fn: Callable[[List[Any]], float]) -> float:
        result = await self.graph.execute(question, sample=True)
        reward = score_fn(result)

        self.baseline = (
            self.baseline_decay * self.baseline
            + (1 - self.baseline_decay) * reward
        )

        advantage = reward - self.baseline
        self._t += 1

        for edge in self.graph.edges:
            self._register_edge(edge)
            key = id(edge)
            p = edge.probability
            g = advantage * (1 - p) / edge.temperature if edge.active else -advantage * p / edge.temperature

            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * g
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * g ** 2

            m_hat = self._m[key] / (1 - self.beta1 ** self._t)
            v_hat = self._v[key] / (1 - self.beta2 ** self._t)

            edge.weight += self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

        return reward

    async def optimize(
        self,
        questions: QuestionList,
        score_fn: Callable[[List[Any]], float] = None,
        n_iterations: int = 100,
        verbose: bool = False,
    ) -> List[float]:
        rewards = []
        for i, item in enumerate(questions[:n_iterations]):
            if isinstance(item, tuple):
                question, q_score_fn = item
            else:
                question, q_score_fn = item, score_fn

            reward = await self.step(question, q_score_fn)
            rewards.append(reward)
            if verbose:
                print(f"Step {i + 1}/{n_iterations}  reward={reward:.2f}  baseline={self.baseline:.2f}")
                for edge in self.graph.edges:
                    print(f"  {repr(edge)}")
        return rewards
