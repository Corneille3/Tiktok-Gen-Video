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
    parser.add_argument("--offline", action="store_true", help="use cache only (no OpenAI calls)")
    parser.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    args = parser.parse_args()

    cert = args.cert.lower().strip()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY (set it in .env).")

    questions = load_questions(cert, data_dir="data")
    batch = select_batch(questions, args.count)

    if not batch:
        print("No questions selected.")
        return

    total_attempted = len(batch)
    skipped = 0
    succeeded = 0
    failed: list[int] = []

    for q in batch:
        qid = q["id"]
        print(f"\n=== Q{qid} ===")

        expected_video = f"output/videos/{cert}_q{qid:03d}.mp4"
        if not args.force and os.path.exists(expected_video):
            print(f"Skipping Q{qid} (video exists): {expected_video}")
            skipped += 1
            continue

        try:
            explanation = generate_explanation(cert, q, model="gpt-4o-mini", offline=args.offline)
            print("Explanation:", explanation)

            audio_path = generate_voice_mp3(
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
                out_dir="output/videos",
            )
            print("Video:", video_path)
            succeeded += 1
        except Exception as e:
            print(f"Failed Q{qid}: {e}")
            failed.append(qid)
            continue

    print("\n=== Run Summary ===")
    print(f"Attempted: {total_attempted}")
    print(f"Skipped: {skipped}")
    print(f"Succeeded: {succeeded}")
    if failed:
        print(f"Failed: {len(failed)} (QIDs: {', '.join(map(str, failed))})")
    else:
        print("Failed: 0")
    print("\nDone.")


if __name__ == "__main__":
    main()
