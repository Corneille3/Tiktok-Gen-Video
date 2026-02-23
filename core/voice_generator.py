import os
from pathlib import Path
from typing import Any, Dict

import boto3
import time


def _ssml_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def generate_voice_mp3(
    cert: str,
    question_obj: Dict[str, Any],
    explanation: str,
    out_dir: str = "output/audio",
) -> str:
    """
    Voice generation (Amazon Polly):
    - Voice: Joanna (neural)
    - Output: MP3
    Read:
    - Question
    - Answers
    - "Comment your answer now."
    - "The correct answer is B."
    - Explanation
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    voice_id = os.getenv("POLLY_VOICE_ID", "Joanna")
    engine = os.getenv("POLLY_ENGINE", "neural")

    polly = boto3.client("polly", region_name=region)

    qid = int(question_obj["id"])
    c = question_obj["choices"]
    correct = question_obj["correct_answer"]
    expl = _ssml_escape(explanation)
    if isinstance(correct, (list, tuple, set)):
        correct_letters = [str(x).strip().upper() for x in correct if str(x).strip()]
    elif isinstance(correct, str):
        if "," in correct:
            correct_letters = [x.strip().upper() for x in correct.split(",") if x.strip()]
        else:
            correct_letters = [correct.strip().upper()]
    else:
        correct_letters = [str(correct).strip().upper()]
    correct_letters = [c for c in correct_letters if c]
    if len(correct_letters) > 1:
        if len(correct_letters) == 2:
            correct_phrase = f"{correct_letters[0]} and {correct_letters[1]}"
        else:
            correct_phrase = ", ".join(correct_letters[:-1]) + f", and {correct_letters[-1]}"
        correct_sentence = f"The correct answers are {correct_phrase}."
    else:
        correct_sentence = f"The correct answer is {correct_letters[0] if correct_letters else 'A'}."

    # SSML pacing for TikTok
    ssml = f"""
<speak>
  <prosody rate="medium">
    Get ready… Question {qid}.
    <break time="12s"/>
    Comment A, B, C, or D.
    <break time="900ms"/>
    {correct_sentence}
    <break time="350ms"/>
    {expl}
  </prosody>
</speak>
""".strip()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{cert}_q{qid:03d}.mp3"

    max_retries = 3
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = polly.synthesize_speech(
                Text=ssml,
                TextType="ssml",
                OutputFormat="mp3",
                VoiceId=voice_id,
                Engine=engine,
            )
            break
        except Exception as e:
            last_err = e
            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"Polly synth failed after {max_retries} attempts "
                    f"(cert={cert}, qid={qid}, region={region}): {e}"
                ) from e
            time.sleep(2**attempt)

    with out_path.open("wb") as f:
        f.write(resp["AudioStream"].read())

    # Speech marks for dynamic subtitles (sentence-level)
    marks_path = Path(f"{out_path}.marks.json")
    last_err = None
    for attempt in range(max_retries):
        try:
            marks_resp = polly.synthesize_speech(
                Text=ssml,
                TextType="ssml",
                OutputFormat="json",
                SpeechMarkTypes=["sentence"],
                VoiceId=voice_id,
                Engine=engine,
            )
            break
        except Exception as e:
            last_err = e
            if attempt >= max_retries - 1:
                raise RuntimeError(
                    f"Polly speech-marks failed after {max_retries} attempts "
                    f"(cert={cert}, qid={qid}, region={region}): {e}"
                ) from e
            time.sleep(2**attempt)

    with marks_path.open("wb") as f:
        f.write(marks_resp["AudioStream"].read())

    return str(out_path)
