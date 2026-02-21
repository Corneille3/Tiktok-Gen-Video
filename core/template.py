import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Pillow 10+ compatibility for MoviePy 1.0.3
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
)

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

@dataclass
class TemplateConfig:
    # Video specs
    width: int = 1080
    height: int = 1920
    fps: int = 30

    # Layout
    safe_width: int = 900
    top_bar_h: int = 120
    card_margin_x: int = 60
    card_top_gap: int = 60
    card_height: int = 1320

    # Timing (seconds)
    hook_start: float = 0.0
    hook_end: float = 2.0

    qa_start: float = 4.0

    engage_start: float = 13.0
    engage_end: float = 16.0

    reveal_start: float = 16.0
    reveal_end: float = 18.0

    explain_start: float = 18.0

    outro_last_seconds: float = 3.0

    # Colors (canonical)
    bg: str = "#0F172A"
    card: str = "#1E293B"
    q_text: str = "#FFFFFF"
    a_text: str = "#CBD5E1"
    ok: str = "#00FF7F"
    accent: str = "#FF9900"

    # Background motion (scale 1.0 -> 1.03)
    bg_zoom_total: float = 0.03

    # Fonts (recommended)
    font_bold: str = "assets/fonts/Poppins-Bold.ttf"
    font_med: str = "assets/fonts/Poppins-Medium.ttf"


class AWSVideoTemplate:
    """
    Premium Visual Template Design (TikTok optimized)
    - 1080x1920, 30 FPS, safe margins
    - Top orange branding bar
    - Card background
    - Question + answers
    - Engagement prompt
    - Answer reveal highlight rectangle with fade-in + slight scale
    - Explanation (2 sentences)
    - Footer branding
    - Subtle background zoom
    """

    def __init__(
        self,
        cert: str,
        day_label: str,
        question_obj: Dict[str, Any],
        explanation: str,
        audio_path: str,
        config: TemplateConfig | None = None,
    ):
        self.cert = cert.upper()
        self.day_label = day_label
        self.q = question_obj
        self.explanation = explanation
        self.audio_path = audio_path
        self.cfg = config or TemplateConfig()

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        self.audio = AudioFileClip(audio_path)
        self.dur = float(self.audio.duration)

    # ----------------
    # Helpers
    # ----------------

    def _font(self, path: str) -> str:
        return path if Path(path).exists() else "DejaVu-Sans"

    def _hex_to_rgba(self, hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)

    def _wrap_lines(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        words = (text or "").split()
        if not words:
            return [""]
        lines, line = [], ""
        for w in words:
            test = (line + " " + w).strip()
            if font.getlength(test) <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
        return lines

    def _rounded_rect_rgba(self, size: Tuple[int, int], radius: int, fill_rgba: Tuple[int, int, int, int]) -> np.ndarray:
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([0, 0, w, h], radius=radius, fill=fill_rgba)
        return np.array(img)

    def _shadowed_text_clip(
        self,
        txt: str,
        fontsize: int,
        color: str,
        font: str,
        max_width: int,
        align: str = "West",
    ) -> ImageClip:
        align_norm = "center" if align.lower() == "center" else "left"
        return self.make_text_clip(
            txt,
            fontsize=fontsize,
            color_hex=color,
            max_width=max_width,
            align=align_norm,
            font_path=font,
            shadow=True,
        )

    def _clip_window(self, start: float, end: float) -> Tuple[float, float]:
        # Clamp to duration (avoid negative durations)
        s = min(max(start, 0.0), self.dur)
        e = min(max(end, 0.0), self.dur)
        return s, max(s, e)

    def _fit_answer_layout(self, texts: List[str], max_width: int, available: float) -> Tuple[int, float, List[int]]:
        # Fit 4 answers into available height by shrinking font size.
        min_size = 36
        for fontsize in range(46, min_size - 1, -1):
            font_path = self._font(self.cfg.font_med)
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except Exception:
                font = ImageFont.load_default()

            pad_y = 6
            ascent, descent = font.getmetrics()
            line_h = ascent + descent + 10

            heights: List[int] = []
            for t in texts:
                lines = self._wrap_lines(t, font, max_width - 16)  # pad_x * 2
                h = pad_y * 2 + line_h * len(lines)
                heights.append(h)

            max_h = max(heights) if heights else 0
            line_gap = max(max_h + 12, 80)
            total = max_h + line_gap * max(len(texts) - 1, 1)
            if total <= available:
                return fontsize, line_gap, heights

        # If still too tall at min size, clamp gap to 0.
        fontsize = min_size
        font_path = self._font(self.cfg.font_med)
        try:
            font = ImageFont.truetype(font_path, fontsize)
        except Exception:
            font = ImageFont.load_default()

        pad_y = 6
        ascent, descent = font.getmetrics()
        line_h = ascent + descent + 10
        heights = []
        for t in texts:
            lines = self._wrap_lines(t, font, max_width - 16)
            h = pad_y * 2 + line_h * len(lines)
            heights.append(h)
        max_h = max(heights) if heights else 0
        line_gap = max(max_h + 12, 80)
        return fontsize, line_gap, heights

    def make_text_clip(
        self,
        text: str,
        fontsize: int,
        color_hex: str,
        max_width: int,
        align: str = "left",   # "left" or "center"
        font_path: str | None = None,
        shadow: bool = True,
    ):
        font_path = font_path or self._font(self.cfg.font_med)
        try:
            font = ImageFont.truetype(font_path, fontsize)
        except Exception:
            font = ImageFont.load_default()

        pad_x, pad_y = 8, 6
        lines = self._wrap_lines(text, font, max_width - 2 * pad_x)

        ascent, descent = font.getmetrics()
        line_h = ascent + descent + 10
        img_h = max(1, pad_y * 2 + line_h * len(lines))

        img = Image.new("RGBA", (max_width, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        fill = self._hex_to_rgba(color_hex, 255)
        shadow_fill = (0, 0, 0, 140)

        y = pad_y
        for ln in lines:
            if align == "center":
                x = (max_width - int(font.getlength(ln))) // 2
            else:
                x = pad_x

            if shadow:
                draw.text((x + 3, y + 3), ln, font=font, fill=shadow_fill)
            draw.text((x, y), ln, font=font, fill=fill)
            y += line_h

        return ImageClip(np.array(img))

    # ----------------
    # Build parts
    # ----------------
    def build_background(self):
        cfg = self.cfg
        bg = ColorClip(size=(cfg.width, cfg.height), color=hex_to_rgb(cfg.bg), duration=self.dur)
        # subtle zoom 1.0 -> 1.03 over duration
        bg = bg.resize(lambda t: 1.0 + cfg.bg_zoom_total * (t / max(self.dur, 0.001)))
        return bg.set_position(("center", "center"))

    def build_top_bar(self) -> List:
        cfg = self.cfg
        bar = ColorClip(size=(cfg.width, cfg.top_bar_h), color=hex_to_rgb(cfg.accent), duration=self.dur).set_position((0, 0))
        title = self.make_text_clip(
            f"AWS {self.cert} • {self.day_label}",
            fontsize=52,
            color_hex="#FFFFFF",
            max_width=cfg.width - 80,
            align="left",
            font_path=self._font(cfg.font_bold),
        ).set_position((40, 30)).set_duration(self.dur)
        return [bar, title]

    def build_card(self):
        cfg = self.cfg
        x = cfg.card_margin_x
        y = cfg.top_bar_h + cfg.card_top_gap
        w = cfg.width - (cfg.card_margin_x * 2)
        h = cfg.card_height
        card = ColorClip(size=(w, h), color=hex_to_rgb(cfg.card), duration=self.dur).set_opacity(0.92)
        return card.set_position((x, y))

    def build_hook(self):
        cfg = self.cfg
        s, e = self._clip_window(cfg.hook_start, cfg.hook_end)
        d = max(e - s, 0.0)
        if d <= 0:
            return None

        hook = self.make_text_clip(
            "🚨 Most People Fail This AWS Question",
            fontsize=64,
            color_hex="#FFFFFF",
            max_width=cfg.safe_width,
            align="center",
            font_path=self._font(cfg.font_bold),
        ).set_position(("center", 320)).set_start(s).set_duration(d).fadeout(0.4)
        return hook

    def build_question(self):
        cfg = self.cfg
        card_y = cfg.top_bar_h + cfg.card_top_gap
        start = cfg.hook_end
        end = cfg.qa_start
        d = max(end - start, 0.1)
        q_clip = self.make_text_clip(
            self.q["question"],
            fontsize=52,
            color_hex=self.cfg.q_text,
            max_width=self.cfg.safe_width,
            align="left",
            font_path=self._font(self.cfg.font_bold),
            shadow=True,
        ).set_position((90, card_y + 60)).set_start(start).set_duration(d).fadein(0.25)
        # Optional overlay hint (recommended for long questions)
        hint_text = "Pause to read • Answers next 👇"
        hint_text_clip = self.make_text_clip(
            hint_text,
            fontsize=40,
            color_hex=self.cfg.accent,
            max_width=self.cfg.safe_width,
            align="center",
            font_path=self._font(self.cfg.font_med),
            shadow=False,
        )
        hint_pad_x, hint_pad_y = 22, 10
        hint_bg = self._rounded_rect_rgba(
            (hint_text_clip.w + hint_pad_x * 2, hint_text_clip.h + hint_pad_y * 2),
            radius=26,
            fill_rgba=(0, 0, 0, int(255 * 0.7)),
        )
        hint_y = (card_y + 60) + q_clip.h + 10
        hint = CompositeVideoClip(
            [
                ImageClip(hint_bg).set_position(("center", hint_y - hint_pad_y)),
                hint_text_clip.set_position(("center", hint_y)),
            ],
            size=(cfg.width, cfg.height),
        ).set_start(start).set_duration(d).set_opacity(0.95)

        return [q_clip, hint]

    def build_answers(self) -> List:
        cfg = self.cfg
        card_y = cfg.top_bar_h + cfg.card_top_gap
        start = cfg.qa_start
        end = cfg.engage_start
        d = max(end - start, 0.1)

        c = self.q["choices"]
        texts = [f"A. {c['A']}", f"B. {c['B']}", f"C. {c['C']}", f"D. {c['D']}"]
        available = (cfg.height - 210) - (card_y + 360)
        fontsize, line_gap, _ = self._fit_answer_layout(texts, cfg.safe_width, available)

        answer_start_y = card_y + 360
        y = answer_start_y
        clips: List = []
        self._answer_positions = {}

        for letter, text in [("A", c["A"]), ("B", c["B"]), ("C", c["C"]), ("D", c["D"])]:
            clip = self.make_text_clip(
                f"{letter}. {text}",
                fontsize=fontsize,
                color_hex=self.cfg.a_text,
                max_width=self.cfg.safe_width,
                align="left",
                font_path=self._font(self.cfg.font_med),
            ).set_position((110, y)).set_start(start).set_duration(d).fadein(0.25)
            clips.append(clip)
            self._answer_positions[letter] = (y, clip.h)
            y += line_gap

        return clips

    def build_engagement_prompt(self):
        cfg = self.cfg
        s, e = self._clip_window(cfg.engage_start, cfg.engage_end)
        d = max(e - s, 0.0)
        if d <= 0:
            return None

        card_y = cfg.top_bar_h + cfg.card_top_gap
        engage = self.make_text_clip(
            "Comment your answer 👇",
            fontsize=54,
            color_hex="#FFFFFF",
            max_width=self.cfg.safe_width,
            align="center",
            font_path=self._font(self.cfg.font_bold),
        )
        engage_y = card_y + (cfg.card_height - engage.h) // 2
        engage = engage.set_position(("center", engage_y)).set_start(s).set_duration(d).fadein(0.35).fadeout(0.35)
        return engage

    def build_reveal(self):
        cfg = self.cfg
        s, e = self._clip_window(cfg.reveal_start, cfg.reveal_end)
        d = max(e - s, 0.0)
        if d <= 0:
            return None

        card_y = cfg.top_bar_h + cfg.card_top_gap
        correct = self.q["correct_answer"]
        hl_x = 90
        hl_w = cfg.width - 180
        hl_h = 90

        if hasattr(self, "_answer_positions") and correct in self._answer_positions:
            y, h = self._answer_positions[correct]
            hl_y = y - 18
            hl_h = h + 36
        else:
            answer_start_y = card_y + 360
            line_gap = 105
            correct_idx = {"A": 0, "B": 1, "C": 2, "D": 3}[correct]
            hl_y = answer_start_y + correct_idx * line_gap - 18

        # green rounded rectangle behind correct answer
        hl_img = self._rounded_rect_rgba((hl_w, hl_h), radius=26, fill_rgba=(0, 255, 127, 55))
        highlight = (
            ImageClip(hl_img, ismask=False)
            .set_position((hl_x, hl_y))
            .set_start(s)
            .set_duration(d)
            .fadein(0.3)
            # slight scale 1.0 -> 1.05 in first second
            .resize(lambda t: 1.0 + 0.05 * min(max(t, 0.0), 1.0))
        )
        return highlight

    def build_explanation(self):
        cfg = self.cfg
        explain_start = cfg.explain_start
        explain_end = max(explain_start, self.dur - cfg.outro_last_seconds)
        s, e = self._clip_window(explain_start, explain_end)
        d = max(e - s, 0.0)
        if d <= 0:
            return None

        card_y = cfg.top_bar_h + cfg.card_top_gap
        exp = self.make_text_clip(
            self.explanation,
            fontsize=44,
            color_hex="#FFFFFF",
            max_width=self.cfg.safe_width,
            align="left",
            font_path=self._font(self.cfg.font_med),
        ).set_position((90, card_y + 60)).set_start(s).set_duration(d).fadein(0.25)
        return exp

    def build_footer(self):
        cfg = self.cfg
        # Footer branding (always visible, plus it serves as "Outro" messaging)
        footer = self.make_text_clip(
            "@yourhandle  •  Follow to pass AWS 🚀",
            fontsize=38,
            color_hex="#FFFFFF",
            max_width=self.cfg.width - 120,
            align="center",
            font_path=self._font(self.cfg.font_med),
            shadow=False,
        ).set_position(("center", self.cfg.height - 90)).set_duration(self.dur).set_opacity(0.9)
        return footer

    def build_audio(self):
        music_path = os.getenv("BACKGROUND_MUSIC_PATH", "").strip()
        music_vol = float(os.getenv("BACKGROUND_MUSIC_VOLUME", "0.06"))
        if music_path and Path(music_path).exists():
            music = AudioFileClip(music_path).volumex(music_vol).set_duration(self.dur)
            return CompositeAudioClip([music, self.audio])
        return self.audio

    # ----------------
    # Compose / Render
    # ----------------
    def compose(self) -> CompositeVideoClip:
        layers: List = []
        layers.append(self.build_background())
        layers.extend(self.build_top_bar())
        layers.append(self.build_card())

        hook = self.build_hook()
        if hook:
            layers.append(hook)

        q_layers = self.build_question()
        layers.extend(q_layers if isinstance(q_layers, list) else [q_layers])
        layers.extend(self.build_answers())

        engage = self.build_engagement_prompt()
        if engage:
            layers.append(engage)

        reveal = self.build_reveal()
        if reveal:
            layers.append(reveal)

        exp = self.build_explanation()
        if exp:
            layers.append(exp)

        layers.append(self.build_footer())

        video = CompositeVideoClip(layers, size=(self.cfg.width, self.cfg.height)).set_audio(self.build_audio())
        return video

    def render(self, out_path: str):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        video = self.compose()
        video.write_videofile(
            out_path,
            fps=self.cfg.fps,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="medium",
        )
        return out_path
