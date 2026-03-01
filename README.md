# AWS TikTok Bot

Generate short-form AWS certification Q&A videos (TikTok-ready) with voice narration, dynamic visuals, and automated overlays.

## Tools Used

- **OpenAI**: Generates short explanations for each question (`core/explanation_generator.py`).
- **Amazon Polly**: Produces the narrated MP3 voiceover (`core/voice_generator.py`).
- **MoviePy + Pillow**: Renders video frames, text overlays, and animations (`core/template.py`, `core/video_renderer.py`).

## Setup

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

2. Create a `.env` file in the project root and set your OpenAI key:

```bash
OPENAI_API_KEY=sk-...
```

3. (Optional) Configure AWS Polly credentials in your environment if you use voice synthesis.

## Usage

Run the generator with:

```bash
./run.sh --cert saa --count 1
```

Common flags:

- `--cert` cert file name in `data/` (e.g. `saa`)
- `--count` number of videos to generate
- `--offline` use cached explanation only (no OpenAI calls)
- `--force` regenerate outputs even if they already exist
