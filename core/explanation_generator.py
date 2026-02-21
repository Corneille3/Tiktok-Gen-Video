import json
import os
import re
import hashlib
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


def _compute_input_hash(question_obj: Dict[str, Any], model: str) -> str:
    payload = {
        "model": model,
        "question": question_obj["question"],
        "choices": {
            "A": question_obj["choices"]["A"],
            "B": question_obj["choices"]["B"],
            "C": question_obj["choices"]["C"],
            "D": question_obj["choices"]["D"],
        },
        "correct_answer": question_obj["correct_answer"],
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def generate_explanation(
    question_obj: Dict[str, Any],
    model: str = "gpt-4o-mini",
    cert: str | None = None,
    offline: bool = False,
) -> str:
    """
    Output must be:
    - 2 short sentences
    - beginner friendly
    - no heavy jargon
    - optimized for ~30s TikTok
    """
    if not cert:
        raise ValueError("Missing cert for explanation cache path.")

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
    correct = question_obj["correct_answer"]
    correct_text = choices[correct]

    prompt = (
        f"Question: {q}\n"
        f"Choices:\n"
        f"A: {choices['A']}\n"
        f"B: {choices['B']}\n"
        f"C: {choices['C']}\n"
        f"D: {choices['D']}\n"
        f"Correct answer: {correct} ({correct_text})\n\n"
        "Explain in 2 short sentences why the correct answer is correct.\n"
        "Keep it beginner friendly.\n"
        "Avoid technical jargon.\n"
        "Suitable for TikTok.\n"
        "Output exactly two sentences."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

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
            "correct_answer": question_obj["correct_answer"],
        },
        "explanation": explanation,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    return explanation
