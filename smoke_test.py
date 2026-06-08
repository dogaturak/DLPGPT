import asyncio
from graph.llm_node import LLMNode


async def main():
    node = LLMNode(
        system_prompt="You are a helpful assistant.",
        model="llama3.1:latest"
    )

    print("Testing Ollama connection...")

    result = await node._execute("Say hello in one short sentence.")

    print("Response:", result)

if __name__ == "__main__":
    asyncio.run(main())
