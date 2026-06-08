import re
from typing import Any, List, Optional

from openai import AsyncOpenAI
from graph.node import Node
from graph.token_tracker import tracker


class LLMNode(Node):
    def __init__(
        self,
        system_prompt: str,
        model: str = "llama3.1",
        operation_description: str = "",
        node_id: Optional[str] = None,
        combine_inputs_as_one: bool = False,
        split_output: bool = False,
    ):
        super().__init__(
            operation_description=operation_description or system_prompt[:40],
            node_id=node_id,
            combine_inputs_as_one=combine_inputs_as_one,
        )

        self.system_prompt = system_prompt
        self.model = model
        self.split_output = split_output

        # ✅ Create client per instance (NO import-time networking)
        self.client = AsyncOpenAI(
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
        )

    async def _execute(self, input: Any, **kwargs) -> Any:
        # ✅ Proper async call with timeout to prevent HPC hangs
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": str(input)},
            ],
            timeout=60,
        )

        # Track token usage if available
        if response.usage:
            tracker.record(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        text = response.choices[0].message.content

        if self.split_output:
            return self._parse_numbered_list(text)

        return text

    @staticmethod
    def _parse_numbered_list(text: str) -> List[str]:
        items = re.findall(r"^\s*\d+[\.\)]\s*(.+)", text, re.MULTILINE)
        return items if items else [text]
