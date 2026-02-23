import json
import os
import re
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


def _enforce_two_sentences(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return "This option best fits the requirement. It provides the right service for the scenario."

    # Split on sentence terminators
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    if len(parts) == 1:
        # Ensure it ends with punctuation then add a simple second sentence
        if parts[0][-1] not in ".!?":
            parts[0] += "."
        return f"{parts[0]} It matches what the question asks for."
    return "This option best fits the requirement. It provides the right service for the scenario."


def _cache_path(cert: str, input_hash: str) -> Path:
    return Path("data/cache/explanations") / cert / f"{input_hash}.json"


def _normalize_correct_answers(correct: Any) -> list[str]:
    if isinstance(correct, (list, tuple, set)):
        items = [str(c).strip().upper() for c in correct if str(c).strip()]
    elif isinstance(correct, str):
        if "," in correct:
            items = [c.strip().upper() for c in correct.split(",") if c.strip()]
        else:
            items = [correct.strip().upper()]
    else:
        items = [str(correct).strip().upper()]
    return [c for c in items if c]


def _format_correct_answers(correct_answers: list[str]) -> str:
    if not correct_answers:
        return "A"
    if len(correct_answers) == 1:
        return correct_answers[0]
    if len(correct_answers) == 2:
        return f"{correct_answers[0]} and {correct_answers[1]}"
    return ", ".join(correct_answers[:-1]) + f", and {correct_answers[-1]}"


def _compute_input_hash(question_obj: Dict[str, Any], model: str) -> str:
    correct_answers = _normalize_correct_answers(question_obj["correct_answer"])
    payload = {
        "model": model,
        "question": question_obj["question"],
        "choices": {
            "A": question_obj["choices"]["A"],
            "B": question_obj["choices"]["B"],
            "C": question_obj["choices"]["C"],
            "D": question_obj["choices"]["D"],
        },
        "correct_answers": correct_answers,
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def explanation_cache_path(cert: str, question_obj: Dict[str, Any], model: str) -> Path:
    input_hash = _compute_input_hash(question_obj, model)
    return _cache_path(cert, input_hash)


def generate_explanation(
    cert: str,
    question_obj: Dict[str, Any],
    model: str = "gpt-4o-mini",
    offline: bool = False,
) -> str:
    """
    Output must be:
    - 2 short sentences
    - beginner friendly
    - no heavy jargon
    - optimized for ~30s TikTok
    """
    input_hash = _compute_input_hash(question_obj, model)
    cache_path = _cache_path(cert, input_hash)

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        return cached.get("explanation", "")

    if offline:
        raise RuntimeError(f"Cache miss in offline mode: {cache_path}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")

    client = OpenAI(api_key=api_key)

    q = question_obj["question"]
    choices = question_obj["choices"]
    correct_answers = _normalize_correct_answers(question_obj["correct_answer"])
    correct_phrase = _format_correct_answers(correct_answers)
    correct_texts = [choices[c] for c in correct_answers if c in choices]

    prompt = (
        f"Question: {q}\n"
        f"Choices:\n"
        f"A: {choices['A']}\n"
        f"B: {choices['B']}\n"
        f"C: {choices['C']}\n"
        f"D: {choices['D']}\n"
        f"Correct answers: {correct_phrase}"
        + (f" ({' / '.join(correct_texts)})" if correct_texts else "")
        + "\n\n"
        "Explain in 2 short sentences why the correct answers are correct.\n"
        "If there are multiple answers, mention both in the explanation.\n"
        "Keep it beginner friendly.\n"
        "Avoid technical jargon.\n"
        "Suitable for TikTok.\n"
        "Output exactly two sentences."
    )

    max_retries = 3
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            break
        except Exception as e:
            last_err = e
            if attempt >= max_retries - 1:
                raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {e}") from e
            time.sleep(2**attempt)

    raw = (resp.choices[0].message.content or "").strip()
    explanation = _enforce_two_sentences(raw)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_hash": input_hash,
        "cert": cert,
        "model": model,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "id": question_obj.get("id"),
            "question": question_obj["question"],
            "choices": {
                "A": question_obj["choices"]["A"],
                "B": question_obj["choices"]["B"],
                "C": question_obj["choices"]["C"],
                "D": question_obj["choices"]["D"],
            },
            "correct_answers": correct_answers,
        },
        "explanation": explanation,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    return explanation
