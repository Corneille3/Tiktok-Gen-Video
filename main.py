import argparse
import os
import json
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

from core.question_loader import load_questions, select_batch
from core.explanation_generator import generate_explanation, explanation_cache_path
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

    state_path = Path("data/state") / f"{cert}.json"
    last_id = 0
    if state_path.exists():
        try:
            with state_path.open("r", encoding="utf-8") as f:
                last_id = int(json.load(f).get("last_id_generated", 0))
        except Exception:
            last_id = 0

    batch = [q for q in questions if int(q.get("id", 0)) > last_id][: args.count]

    if not batch:
        print("No questions selected.")
        return

    total_attempted = len(batch)
    skipped = 0
    succeeded = 0
    failed: list[int] = []

    run_date = date.today().isoformat()
    batch_rows: list[dict[str, str]] = []

    for q in batch:
        qid = q["id"]
        print(f"\n=== Q{qid} ===")

        expected_video = f"output/videos/{cert}_q{qid:03d}.mp4"
        if not args.force and os.path.exists(expected_video):
            print(f"Skipping Q{qid} (video exists): {expected_video}")
            skipped += 1
            cache_hit = explanation_cache_path(cert, q, model="gpt-4o-mini").exists()
            batch_rows.append(
                {
                    "date": run_date,
                    "cert": cert,
                    "qid": str(qid),
                    "video_path": expected_video,
                    "audio_path": "",
                    "cache_hit": "true" if cache_hit else "false",
                    "status": "skipped",
                }
            )
            last_id = max(last_id, int(qid))
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with state_path.open("w", encoding="utf-8") as f:
                json.dump({"last_id_generated": last_id}, f, ensure_ascii=True, indent=2)
            continue

        try:
            cache_hit = explanation_cache_path(cert, q, model="gpt-4o-mini").exists()
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
            batch_rows.append(
                {
                    "date": run_date,
                    "cert": cert,
                    "qid": str(qid),
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "cache_hit": "true" if cache_hit else "false",
                    "status": "generated",
                }
            )
            last_id = max(last_id, int(qid))
            state_path.parent.mkdir(parents=True, exist_ok=True)
            with state_path.open("w", encoding="utf-8") as f:
                json.dump({"last_id_generated": last_id}, f, ensure_ascii=True, indent=2)
        except Exception as e:
            print(f"Failed Q{qid}: {e}")
            failed.append(qid)
            batch_rows.append(
                {
                    "date": run_date,
                    "cert": cert,
                    "qid": str(qid),
                    "video_path": "",
                    "audio_path": "",
                    "cache_hit": "false",
                    "status": "failed",
                }
            )
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

    # Write batch CSV
    if batch_rows:
        out_dir = Path("output/batches")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cert}_{run_date}.csv"
        header = ["date", "cert", "qid", "video_path", "audio_path", "cache_hit", "status"]
        write_header = not out_path.exists()
        with out_path.open("a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(header) + "\n")
            for row in batch_rows:
                f.write(",".join(row[h] for h in header) + "\n")


if __name__ == "__main__":
    main()
