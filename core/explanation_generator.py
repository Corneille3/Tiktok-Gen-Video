import os
import re
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


def generate_explanation(question_obj: Dict[str, Any], model: str = "gpt-4o-mini") -> str:
    """
    Output must be:
    - 2 short sentences
    - beginner friendly
    - no heavy jargon
    - optimized for ~30s TikTok
    """
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
    return _enforce_two_sentences(raw)