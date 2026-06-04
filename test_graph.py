import asyncio
from graph import LLMNode, Edge, Graph


async def main():
    decomposer = LLMNode(
        "Break this question into exactly 2 sub-questions. Return them as a numbered list, one per line.",
        split_output=True,
        operation_description="Decomposer",
    )
    solver = LLMNode(
        "You are given a sub-question. Answer it in exactly one sentence using only factual information.",
        operation_description="Solver",
    )
    aggregator = LLMNode(
        "You are given multiple answers. Combine them into one clear, concise final answer.",
        combine_inputs_as_one=True,
        operation_description="Aggregator",
    )

    graph = Graph(output_node=aggregator)
    graph.add_edge(Edge(decomposer, solver))
    graph.add_edge(Edge(solver, aggregator))

    graph.describe()

    question = "What causes inflation and how does it affect employment?"
    print(f"Question: {question}\n")

    result = await graph.execute(question, verbose=True)
    print(f"\nFinal Answer: {result[0]}")


asyncio.run(main())
