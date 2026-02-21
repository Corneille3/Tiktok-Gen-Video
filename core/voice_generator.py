import os
from pathlib import Path
from typing import Any, Dict

import boto3


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
    q = _ssml_escape(question_obj["question"])
    c = question_obj["choices"]
    correct = question_obj["correct_answer"]
    expl = _ssml_escape(explanation)

    # SSML pacing for TikTok
    ssml = f"""
<speak>
  <prosody rate="medium">
    Question {qid}. {q}
    <break time="800ms"/>
    Comment A, B, C, or D.
    <break time="900ms"/>
    The correct answer is {correct}.
    <break time="350ms"/>
    {expl}
  </prosody>
</speak>
""".strip()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{cert}_q{qid:03d}.mp3"

    resp = polly.synthesize_speech(
        Text=ssml,
        TextType="ssml",
        OutputFormat="mp3",
        VoiceId=voice_id,
        Engine=engine,
    )

    with out_path.open("wb") as f:
        f.write(resp["AudioStream"].read())

    return str(out_path)