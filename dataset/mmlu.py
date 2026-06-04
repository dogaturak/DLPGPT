import re
from typing import Any, Dict, List, Tuple

ANSWER_INDEX = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_mmlu(subject: str = "all", split: str = "test", n: int = 100) -> List[Dict]:
    """Load n questions from MMLU. subject='all' samples across all subjects."""
    from datasets import load_dataset

    if subject == "all":
        dataset = load_dataset("cais/mmlu", "all", split=split)
    else:
        dataset = load_dataset("cais/mmlu", subject, split=split)

    return [dataset[i] for i in range(min(n, len(dataset)))]


def format_question(item: Dict) -> str:
    """Format an MMLU item into a prompt string."""
    choices = item["choices"]
    lines = [f"Question: {item['question']}"]
    for letter, choice in zip("ABCD", choices):
        lines.append(f"{letter}) {choice}")
    return "\n".join(lines)


def extract_answer(output: str) -> str:
    """Extract the first A/B/C/D letter from model output."""
    match = re.search(r"\b([A-D])\b", output.upper())
    return match.group(1) if match else ""


def score_fn(correct_letter: str):
    """Return a scoring function that checks if output matches correct_letter."""
    def _score(outputs: List[Any]) -> float:
        if not outputs:
            return 0.0
        answer = extract_answer(str(outputs[0]))
        return 1.0 if answer == correct_letter else 0.0
    return _score


def make_questions(items: List[Dict]) -> List[Tuple[str, Any]]:
    """Return list of (formatted_question, score_fn) pairs."""
    return [
        (format_question(item), score_fn(ANSWER_INDEX[item["answer"]]))
        for item in items
    ]
