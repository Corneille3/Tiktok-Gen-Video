import argparse
import os
from dotenv import load_dotenv

from core.question_loader import load_questions, select_batch
from core.explanation_generator import generate_explanation
from core.voice_generator import generate_voice_mp3
from core.video_renderer import render_video


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="AWS TikTok Bot (Class-based Template MVP)")
    parser.add_argument("--cert", required=True, help="cert file name in /data (e.g. saa)")
    parser.add_argument("--count", type=int, default=1, help="how many videos to generate")
    parser.add_argument("--day", default="Day 1", help='label shown in top bar (e.g. "Day 7")')
    args = parser.parse_args()

    cert = args.cert.lower().strip()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY (set it in .env).")

    questions = load_questions(cert, data_dir="data")
    batch = select_batch(questions, args.count)

    if not batch:
        print("No questions selected.")
        return

    for q in batch:
        qid = q["id"]
        print(f"\n=== Q{qid} ===")

        explanation = generate_explanation(q, model="gpt-4o-mini")
        print("Explanation:", explanation)

        audio_path, timeline = generate_voice_mp3(
            cert=cert,
            question_obj=q,
            explanation=explanation,
            out_dir="output/audio",
        )
        print("Audio:", audio_path)

        video_path = render_video(
            cert=cert,
            day_label=args.day,
            question_obj=q,
            explanation=explanation,
            audio_path=audio_path,
            timeline=timeline,
            out_dir="output/videos",
        )
        print("Video:", video_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
