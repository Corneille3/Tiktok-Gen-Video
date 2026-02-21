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

    qa_start: float = 2.0

    engage_start: float = 15.0
    engage_end: float = 18.0

    reveal_start: float = 18.0
    reveal_end: float = 22.0

    explain_start: float = 22.0
    explain_end: float = 35.0

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
        start = cfg.qa_start               # 2s
        end = cfg.engage_start             # 15s
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
        hint = self.make_text_clip(
            "Pause to read • Answers next 👇",
            fontsize=40,
            color_hex="#FFFFFF",
            max_width=self.cfg.safe_width,
            align="center",
            font_path=self._font(self.cfg.font_med),
            shadow=False,
        ).set_position(("center", card_y + 980)).set_start(start).set_duration(d).set_opacity(0.9)

        return [q_clip, hint]

    def build_answers(self) -> List:
        cfg = self.cfg
        card_y = cfg.top_bar_h + cfg.card_top_gap
        start = cfg.engage_start           # 15s
        end = cfg.reveal_start             # 18s
        d = max(end - start, 0.1)

        answer_start_y = card_y + 360      # fixed now (no need dynamic question height)
        line_gap = 105

        def make(letter: str, text: str, idx: int):
            return self.make_text_clip(
                f"{letter}. {text}",
                fontsize=46,
                color_hex=self.cfg.a_text,
                max_width=self.cfg.safe_width,
                align="left",
                font_path=self._font(self.cfg.font_med),
            ).set_position((110, answer_start_y + idx * line_gap)).set_start(start).set_duration(d).fadein(0.25)

        c = self.q["choices"]
        return [
            make("A", c["A"], 0),
            make("B", c["B"], 1),
            make("C", c["C"], 2),
            make("D", c["D"], 3),
        ]

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
        ).set_position(("center", card_y + 980)).set_start(s).set_duration(d).fadein(0.2).fadeout(0.2)
        return engage

    def build_reveal(self):
        cfg = self.cfg
        s, e = self._clip_window(cfg.reveal_start, cfg.reveal_end)
        d = max(e - s, 0.0)
        if d <= 0:
            return None

        card_y = cfg.top_bar_h + cfg.card_top_gap
        answer_start_y = card_y + 360
        line_gap = 105

        correct = self.q["correct_answer"]
        correct_idx = {"A": 0, "B": 1, "C": 2, "D": 3}[correct]

        hl_x = 90
        hl_y = answer_start_y + correct_idx * line_gap - 18
        hl_w = cfg.width - 180
        hl_h = 90

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
        s, e = self._clip_window(cfg.explain_start, cfg.explain_end)
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
        ).set_position((110, card_y + 1080)).set_start(s).set_duration(d).fadein(0.25)
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
