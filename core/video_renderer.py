from typing import Any, Dict

from core.template import AWSVideoTemplate, TemplateConfig


def render_video(
    cert: str,
    day_label: str,
    question_obj: Dict[str, Any],
    explanation: str,
    audio_path: str,
    out_dir: str = "output/videos",
) -> str:
    qid = int(question_obj["id"])
    out_path = f"{out_dir}/{cert}_q{qid:03d}.mp4"

    template = AWSVideoTemplate(
        cert=cert,
        day_label=day_label,
        question_obj=question_obj,
        explanation=explanation,
        audio_path=audio_path,
        config=TemplateConfig(),
    )
    return template.render(out_path)
