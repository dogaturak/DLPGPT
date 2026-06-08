import asyncio
from graph import LLMNode, Graph


async def smoke_test():
    print("Testing LLMNode connection to Ollama...")
    node = LLMNode(
        system_prompt="Respond with only the word 'pong'.",
        model="llama3.1",
        operation_description="SmokeTestNode"
    )
    graph = Graph(output_node=node)
    try:
        result = await graph.execute("ping", sample=False)
        print(f"Success! Model responded: {result}")
    except Exception as e:
        print(f"Failed! Error: {e}")

if __name__ == "__main__":
    asyncio.run(smoke_test())
