import re
import json
import asyncio
import urllib.request
from typing import Any, List, Optional

from graph.node import Node
from graph.token_tracker import tracker


class LLMNode(Node):
    def __init__(
        self,
        system_prompt: str,
        model: str = "llama3.1:latest",
        operation_description: str = "",
        node_id: Optional[str] = None,
        combine_inputs_as_one: bool = False,
        split_output: bool = False,
        ollama_host: str = "http://127.0.0.1:11434",
    ):
        super().__init__(
            operation_description=operation_description or system_prompt[:40],
            node_id=node_id,
            combine_inputs_as_one=combine_inputs_as_one,
        )

        self.system_prompt = system_prompt
        self.model = model
        self.split_output = split_output
        self.url = f"{ollama_host}/api/chat"

    async def _execute(self, input: Any, **kwargs) -> Any:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": str(input)},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }

        try:
            data = await asyncio.wait_for(
                asyncio.to_thread(self._post, payload),
                timeout=600,
            )
        except asyncio.TimeoutError:
            raise RuntimeError("Ollama request timed out (600s)")

        # SAFE PARSING
        text = ""
        if isinstance(data, dict):
            msg = data.get("message", {})
            if isinstance(msg, dict):
                text = msg.get("content", "")

        # TOKEN TRACKING
        if isinstance(data, dict):
            try:
                tracker.record(
                    data.get("prompt_eval_count", 0),
                    data.get("eval_count", 0),
                )
            except Exception:
                pass

        return self._parse_numbered_list(text) if self.split_output else text

    def _post(self, payload):
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _parse_numbered_list(text: str) -> List[str]:
        items = re.findall(r"^\s*\d+[\.\)]\s*(.+)", text, re.MULTILINE)
        return items if items else [text]
