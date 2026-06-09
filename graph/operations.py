import pathlib
import re
from collections import Counter
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI

from graph.node import Node
from graph.token_tracker import tracker

_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

_COMBINE_PROMPT = (
    "You are given multiple pieces of information from different sources. "
    "Synthesize them into a single, coherent, and concise answer."
)


"""class WebSearchNode(Node):
    Generates a focused search query via LLM, then fetches DuckDuckGo results.

    def __init__(
        self,
        model: str = "llama3.1",
        max_results: int = 3,
        operation_description: str = "WebSearch",
        node_id: Optional[str] = None,
    ):
        super().__init__(operation_description=operation_description, node_id=node_id)
        self.model = model
        self.max_results = max_results

    async def _execute(self, input: Any, **kwargs) -> str:
        response = await _client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a short, focused web search query for the following task. "
                        "Output only the query, nothing else."
                    ),
                },
                {"role": "user", "content": str(input)},
            ],
        )
        query = response.choices[0].message.content.strip()

        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            )
        data = resp.json()

        snippets = []
        if data.get("AbstractText"):
            snippets.append(data["AbstractText"])
        for result in data.get("RelatedTopics", [])[: self.max_results]:
            if isinstance(result, dict) and result.get("Text"):
                snippets.append(result["Text"])

        if not snippets:
            return f"No results found for query: {query}"
        return "Search query: " + query + "\n\nResults:\n" + "\n\n".join(snippets)
"""

class FileAnalyzerNode(Node):
    """Reads a file (PDF or text) and uses an LLM to extract relevant information."""

    def __init__(
        self,
        task_prompt: str = "Summarize the key information in this document.",
        model: str = "llama3.1",
        max_chars: int = 4000,
        operation_description: str = "FileAnalyzer",
        node_id: Optional[str] = None,
    ):
        super().__init__(operation_description=operation_description, node_id=node_id)
        self.task_prompt = task_prompt
        self.model = model
        self.max_chars = max_chars

    def _read_file(self, path: str) -> str:
        p = pathlib.Path(path)
        if p.suffix.lower() == ".pdf":
            import fitz  # pymupdf
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        return p.read_text(encoding="utf-8", errors="ignore")

    async def _execute(self, input: Any, **kwargs) -> str:
        """input: a file path string."""
        content = self._read_file(str(input))[: self.max_chars]
        response = await _client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.task_prompt},
                {"role": "user", "content": content},
            ],
        )
        if response.usage:
            tracker.record(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content


class CombineAnswerNode(Node):
    """Aggregates all predecessor outputs into one coherent answer via LLM."""

    def __init__(
        self,
        system_prompt: str = _COMBINE_PROMPT,
        model: str = "llama3.1",
        operation_description: str = "CombineAnswer",
        node_id: Optional[str] = None,
    ):
        super().__init__(
            operation_description=operation_description,
            node_id=node_id,
            combine_inputs_as_one=True,
        )
        self.system_prompt = system_prompt
        self.model = model

    async def _execute(self, input: Any, **kwargs) -> str:
        if isinstance(input, list):
            combined = "\n\n".join(f"Source {i + 1}:\n{item}" for i, item in enumerate(input))
        else:
            combined = str(input)
        response = await _client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": combined},
            ],
        )
        if response.usage:
            tracker.record(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content


class MajorityVoteNode(Node):
    """Deterministic majority vote over predecessor outputs — no LLM call."""

    def __init__(self, operation_description: str = "MajorityVote", node_id: Optional[str] = None):
        super().__init__(
            operation_description=operation_description,
            node_id=node_id,
            combine_inputs_as_one=True,
        )

    async def _execute(self, input: Any, **kwargs) -> str:
        items = input if isinstance(input, list) else [input]
        answers = []
        for item in items:
            m = re.search(r"\b([A-D])\b", str(item).upper())
            if m:
                answers.append(m.group(1))
        if not answers:
            return ""
        return Counter(answers).most_common(1)[0][0]