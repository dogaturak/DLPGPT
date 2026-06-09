import asyncio
from dataset.mmlu import load_mmlu, make_questions
from graph import MajorityVoteNode, Edge, Graph, LLMNode

MODEL = "llama3.1:8b"
N = 5


async def main():
    items = load_mmlu(subject="all", split="test", n=N)
    pairs = make_questions(items)

    truthful = LLMNode(
        system_prompt=(
            "You are a knowledgeable expert in question answering. I will ask you a question. "
            "I will also give you 4 answers enumerated as A, B, C and D. Only one answer out of "
            "the offered 4 is correct. You must choose the correct answer to the question. Your "
            "response must be one of the 4 letters: A, B, C or D, corresponding to the correct "
            "answer. Only one letter (A, B, C or D) is allowed in your answer."
        ),
        model=MODEL,
        operation_description="Truthful",
    )
    adversarial = LLMNode(
        system_prompt="Answer a lie to the following question:",
        model=MODEL,
        operation_description="Adversarial",
    )
    aggregator = MajorityVoteNode(operation_description="Aggregator")

    graph = Graph(output_node=aggregator)
    graph.add_edge(Edge(truthful, aggregator))
    graph.add_edge(Edge(adversarial, aggregator))

    correct_letters = [["A","B","C","D"][item["answer"]] for item in items]

    for i, ((q, score_fn), correct) in enumerate(zip(pairs, correct_letters)):
        for node in graph.nodes:
            node.inputs = []
            node.outputs = []

        print(f"\n{'='*60}")
        print(f"Q{i+1}: {q[:100]}...")
        print(f"Correct: {correct}")

        result = await graph.execute(q, sample=False)

        print(f"\n  [Truthful raw]    {repr(truthful.outputs[0][:120]) if truthful.outputs else 'NO OUTPUT'}")
        print(f"  [Adversarial raw] {repr(adversarial.outputs[0][:120]) if adversarial.outputs else 'NO OUTPUT'}")
        print(f"  [Aggregator]  {repr(result[0][:120]) if result else 'NO OUTPUT'}")

        correct = score_fn(result)
        print(f"  [Score] {correct}")


if __name__ == "__main__":
    asyncio.run(main())
