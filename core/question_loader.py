import json
from pathlib import Path
from typing import Any, Dict, List


def load_questions(cert: str, data_dir: str = "data") -> List[Dict[str, Any]]:
    path = Path(data_dir) / f"{cert}.json"
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of question objects.")

    for q in data:
        for key in ("id", "question", "choices", "correct_answer"):
            if key not in q:
                raise ValueError(f"Question missing '{key}': {q}")

        if not isinstance(q["choices"], dict):
            raise ValueError(f"'choices' must be an object/dict: {q}")

        if q["correct_answer"] not in q["choices"]:
            raise ValueError(f"correct_answer must be one of choices keys: {q}")

        for letter in ("A", "B", "C", "D"):
            if letter not in q["choices"]:
                raise ValueError(f"Missing choice '{letter}' in: {q}")

    return data


def select_batch(questions: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    n = max(0, int(count))
    return questions[:n]