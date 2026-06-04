import random
from typing import Any, Callable, List, Tuple

from graph.graph import Graph


class EASwarm:
    """Evolutionary algorithm optimizer for graph edge topologies.

    Each individual is a binary vector over graph.edges (1=active, 0=inactive).
    Uses tournament selection, single-point crossover, bit-flip mutation,
    and elitism (best individual always survives).
    """

    def __init__(
        self,
        graph: Graph,
        pop_size: int = 8,
        mutation_rate: float = 0.2,
        tournament_k: int = 2,
    ):
        self.graph = graph
        self.n_edges = len(graph.edges)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.population: List[List[int]] = [
            [random.randint(0, 1) for _ in range(self.n_edges)]
            for _ in range(pop_size)
        ]

    def _apply(self, individual: List[int]) -> None:
        for edge, active in zip(self.graph.edges, individual):
            edge.active = bool(active)

    async def _fitness(
        self,
        individual: List[int],
        questions: List[Any],
        score_fns: List[Callable],
    ) -> float:
        self._apply(individual)
        total = 0.0
        for q, sfn in zip(questions, score_fns):
            result = await self.graph.execute(q, sample=False)
            total += sfn(result)
        return total / len(questions) if questions else 0.0

    def _tournament_select(self, fitnesses: List[float]) -> List[int]:
        candidates = random.sample(range(self.pop_size), self.tournament_k)
        winner = max(candidates, key=lambda i: fitnesses[i])
        return self.population[winner][:]

    def _crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        if self.n_edges < 2:
            return p1[:], p2[:]
        point = random.randint(1, self.n_edges - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    def _mutate(self, individual: List[int]) -> List[int]:
        return [bit ^ 1 if random.random() < self.mutation_rate else bit for bit in individual]

    async def optimize(
        self,
        questions: List[Any],
        score_fns: List[Callable],
        n_generations: int = 10,
        batch_size: int = 5,
        verbose: bool = False,
    ) -> Tuple[List[int], List[float]]:
        best_individual = self.population[0][:]
        best_fitness_overall = 0.0
        gen_bests: List[float] = []

        for gen in range(n_generations):
            indices = random.sample(range(len(questions)), min(batch_size, len(questions)))
            batch_q = [questions[i] for i in indices]
            batch_s = [score_fns[i] for i in indices]

            fitnesses = [
                await self._fitness(ind, batch_q, batch_s)
                for ind in self.population
            ]

            best_idx = max(range(self.pop_size), key=lambda i: fitnesses[i])
            gen_best = fitnesses[best_idx]
            gen_bests.append(gen_best)

            if gen_best > best_fitness_overall:
                best_fitness_overall = gen_best
                best_individual = self.population[best_idx][:]

            if verbose:
                mean_f = sum(fitnesses) / len(fitnesses)
                print(
                    f"Gen {gen + 1}/{n_generations}  "
                    f"best={gen_best:.2f}  mean={mean_f:.2f}  "
                    f"topology={self.population[best_idx]}"
                )

            # elitism + tournament selection + crossover + mutation
            new_pop = [self.population[best_idx][:]]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(fitnesses)
                p2 = self._tournament_select(fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            self.population = new_pop

        return best_individual, gen_bests
