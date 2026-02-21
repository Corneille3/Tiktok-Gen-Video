import os
from pathlib import Path
from typing import Any, Dict, Tuple

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
) -> Tuple[str, Dict[str, float | None]]:
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

    # SSML pacing for TikTok
    ssml = f"""
<speak>
  <prosody rate="medium">
    Get ready… Question {qid}.
    <break time="12s"/>
    Comment A, B, C, or D.
    <break time="900ms"/>
    The correct answer is {correct}.
    <break time="350ms"/>
    {expl}
  </prosody>
</speak>
""".strip()

    # Timeline markers (seconds). Some values depend on final audio duration.
    timeline = {
        "t_hook_start": 0.0,
        "t_hook_end": 2.4,
        "t_read_start": 2.6,
        "t_read_end": 14.6,          # read_start + 12.0
        "t_comment_start": 14.6,
        "t_comment_end": 16.6,       # comment_start + 2.0
        "t_reveal_start": 16.6,
        "t_reveal_end": 18.6,        # reveal_start + 2.0
        "t_explain_start": 18.6,
        # t_outro_start and t_end require audio duration; filled after synth.
        "t_outro_start": None,
        "t_end": None,
    }

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

    # Fill timeline markers that depend on the audio duration.
    try:
        from moviepy.editor import AudioFileClip

        audio = AudioFileClip(str(out_path))
        audio_dur = float(audio.duration)
        audio.close()
    except Exception:
        audio_dur = None

    if audio_dur is not None:
        timeline["t_outro_start"] = max(timeline["t_explain_start"], audio_dur - 3.0)
        timeline["t_end"] = audio_dur

    return str(out_path), timeline
