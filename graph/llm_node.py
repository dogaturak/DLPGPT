import re
from typing import Any, List, Optional
from openai import AsyncOpenAI
from graph.node import Node

_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


class LLMNode(Node):
    def __init__(
        self,
        system_prompt: str,
        model: str = "llama3.2",
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

    async def _execute(self, input: Any, **kwargs) -> Any:
        response = await _client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": str(input)},
            ],
        )
        text = response.choices[0].message.content
        if self.split_output:
            return self._parse_numbered_list(text)
        return text

    @staticmethod
    def _parse_numbered_list(text: str) -> List[str]:
        items = re.findall(r"^\s*\d+[\.\)]\s*(.+)", text, re.MULTILINE)
        return items if items else [text]
