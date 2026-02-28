import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import math

# Pillow 10+ compatibility for MoviePy 1.0.3
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
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

    qa_start: float = 5.0

    engage_start: float = 12.0
    engage_end: float = 15.0

    reveal_start: float = 15.0
    reveal_end: float = 17.0

    explain_start: float = 17.0

    outro_last_seconds: float = 0.0

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

    def _countdown_ring_clip(
        self,
        radius: int,
        thickness: int,
        start: float,
        end: float,
        color_hex: str,
    ) -> VideoClip:
        duration = max(end - start, 0.0)
        if duration <= 0:
            return VideoClip(lambda t: np.zeros((1, 1, 4), dtype=np.uint8), duration=0)

        color = self._hex_to_rgba(color_hex, 255)
        size = radius * 2 + thickness * 2

        def make_rgba(t: float) -> np.ndarray:
            frac = 1.0 - min(max(t / duration, 0.0), 1.0)
            end_angle = 360.0 * frac
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            bbox = [thickness, thickness, size - thickness, size - thickness]
            draw.arc(bbox, start=-90, end=-90 + end_angle, fill=color, width=thickness)
            return np.array(img)

        def make_frame(t: float) -> np.ndarray:
            rgba = make_rgba(t)
            return rgba[:, :, :3]

        def make_mask(t: float) -> np.ndarray:
            rgba = make_rgba(t)
            return rgba[:, :, 3] / 255.0

        clip = VideoClip(make_frame=make_frame, duration=duration)
        mask = VideoClip(make_frame=make_mask, duration=duration, ismask=True)
        return clip.set_mask(mask).set_start(start)
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
        # Fit answers into available height by shrinking font size.
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
        bar = ColorClip(size=(cfg.width, cfg.top_bar_h), color=hex_to_rgb(cfg.accent), duration=self.dur)
        bar = bar.set_position(lambda t: (0, -cfg.top_bar_h + min(max(t, 0.0), 0.4) / 0.4 * cfg.top_bar_h))
        episode = int(self.q["id"])
        videos_per_day = 3
        days_total = 30
        day_number = (episode + videos_per_day - 1) // videos_per_day
        if day_number > days_total:
            day_number = days_total
        video_of_day = ((episode - 1) % videos_per_day) + 1
        day_label = f"Day {day_number}/{days_total} • Video {video_of_day}/{videos_per_day}"
        top_font_size = 46

        def _truncate_to_px(text: str, font: ImageFont.FreeTypeFont, max_px: int) -> str:
            if not text:
                return ""
            if font.getlength(text) <= max_px:
                return text
            ell = "…"
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi) // 2
                candidate = text[:mid].rstrip() + ell
                if font.getlength(candidate) <= max_px:
                    lo = mid + 1
                else:
                    hi = mid
            final = text[: max(lo - 1, 0)].rstrip() + ell
            return final if final != ell else ""

        domain_name = str(self.q.get("domain_name", "")).strip()
        domain_suffix = ""
        if domain_name:
            try:
                title_font = ImageFont.truetype(self._font(cfg.font_bold), size=top_font_size)
            except Exception:
                title_font = ImageFont.load_default()
            max_domain_px = 420
            trimmed = _truncate_to_px(domain_name, title_font, max_domain_px)
            if trimmed:
                domain_suffix = f" - {trimmed}"

        top_label = f"AWS {self.cert} • {day_label}{domain_suffix}"
        max_top_px = cfg.width - 96
        font_size = top_font_size
        try:
            title_font = ImageFont.truetype(self._font(cfg.font_bold), size=font_size)
        except Exception:
            title_font = ImageFont.load_default()
        if title_font.getlength(top_label) > max_top_px:
            min_size = 38
            while font_size > min_size:
                font_size -= 2
                try:
                    title_font = ImageFont.truetype(self._font(cfg.font_bold), size=font_size)
                except Exception:
                    title_font = ImageFont.load_default()
                if title_font.getlength(top_label) <= max_top_px:
                    break
            if title_font.getlength(top_label) > max_top_px:
                top_label = _truncate_to_px(top_label, title_font, max_top_px)
        title = self.make_text_clip(
            top_label,
            fontsize=font_size,
            color_hex="#FFFFFF",
            max_width=cfg.width - 80,
            align="left",
            font_path=self._font(cfg.font_bold),
        )
        title = title.set_position(lambda t: (40, -cfg.top_bar_h + 30 + min(max(t, 0.0), 0.4) / 0.4 * cfg.top_bar_h)).set_duration(self.dur)
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

        hook_text = (self.q.get("hook") or "").strip()
        if not hook_text:
            hook_text = "🚨 Most People Fail This AWS Question"

        hook = self.make_text_clip(
            hook_text,
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
        q_clip = q_clip.resize(lambda t: 0.98 + 0.02 * min(max(t, 0.0), 0.3) / 0.3)
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
        order = [letter for letter in ["A", "B", "C", "D", "E"] if letter in c]
        texts = [f"{letter}. {c[letter]}" for letter in order]
        card_bottom = card_y + cfg.card_height
        answer_start_y = card_y + 360
        available = (card_bottom - 80) - answer_start_y
        fontsize, line_gap, _ = self._fit_answer_layout(texts, cfg.safe_width, available)

        y = answer_start_y
        clips: List = []
        self._answer_positions = {}

        stagger = 0.18
        for i, letter in enumerate(order):
            text = c[letter]
            s = start + i * stagger
            clip = self.make_text_clip(
                f"{letter}. {text}",
                fontsize=fontsize,
                color_hex=self.cfg.a_text,
                max_width=self.cfg.safe_width,
                align="left",
                font_path=self._font(self.cfg.font_med),
            ).set_position((110, y)).set_start(s).set_duration(max(0.1, end - s)).fadein(0.25)
            # Subtle background strip for readability
            strip_w = cfg.safe_width + 20
            strip_h = clip.h + 8
            strip_img = self._rounded_rect_rgba((strip_w, strip_h), radius=18, fill_rgba=(0, 0, 0, 80))
            strip = (
                ImageClip(strip_img)
                .set_position((100, y - 4))
                .set_start(s)
                .set_duration(max(0.1, end - s))
                .set_opacity(0.5)
            )
            clips.append(strip)
            clips.append(clip)
            self._answer_positions[letter] = (y, clip.h)
            y += line_gap

        self._answer_fontsize = fontsize
        return clips

    def build_countdown(self) -> List:
        cfg = self.cfg
        card_y = cfg.top_bar_h + cfg.card_top_gap
        answer_start_y = card_y + 360
        y = answer_start_y - 90

        countdown_start = 7.0
        countdown_end = 12.0
        if countdown_end <= countdown_start:
            return []

        label = self.make_text_clip(
            "Time left",
            fontsize=32,
            color_hex="#FFFFFF",
            max_width=cfg.safe_width,
            align="center",
            font_path=self._font(cfg.font_med),
            shadow=True,
        ).set_position(("center", y - 50)).set_start(countdown_start).set_duration(countdown_end - countdown_start).set_opacity(0.7)

        clips: List = [label]
        ring_radius = 40
        ring_thickness = 4
        ring = self._countdown_ring_clip(
            radius=ring_radius,
            thickness=ring_thickness,
            start=countdown_start,
            end=countdown_end,
            color_hex=cfg.accent,
        ).set_position((cfg.width // 2 - ring_radius - ring_thickness, y - ring_thickness))
        clips.append(ring)
        for i, num in enumerate([5, 4, 3, 2, 1]):
            s = countdown_start + i
            e = s + 1.0
            clip = self.make_text_clip(
                str(num),
                fontsize=64,
                color_hex=cfg.accent,
                max_width=cfg.safe_width,
                align="center",
                font_path=self._font(cfg.font_bold),
                shadow=True,
            ).set_position(("center", y)).set_start(s).set_duration(1.0).fadein(0.15).fadeout(0.15)
            clip = clip.resize(lambda t: 1.0 + 0.06 * min(max(t, 0.0), 0.2) / 0.2)
            clips.append(clip)

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
        engage = engage.set_position(("center", engage_y)).set_duration(d)

        icon_path = Path("assets/icons/tap.png")
        if icon_path.exists():
            icon = ImageClip(str(icon_path)).resize(width=64)
        else:
            icon = self.make_text_clip(
                "👇",
                fontsize=64,
                color_hex="#FFFFFF",
                max_width=120,
                align="center",
                font_path=self._font(self.cfg.font_bold),
                shadow=False,
            )
        icon_x = (cfg.width - icon.w) // 2
        icon_base_y = engage_y + engage.h + 8
        icon = icon.set_position(lambda t: (icon_x, icon_base_y + 6 * math.sin(2 * math.pi * 1.2 * t))).set_duration(d)

        composite = CompositeVideoClip(
            [engage, icon],
            size=(cfg.width, cfg.height),
            bg_color=None,
        ).set_start(s).set_duration(d).fadein(0.35).fadeout(0.35)
        return composite

    def build_reveal(self):
        cfg = self.cfg
        reveal_len = max(cfg.reveal_end - cfg.reveal_start, 0.0)
        mark_time = self._find_sentence_time("the correct answer is")
        if mark_time is not None:
            s, e = self._clip_window(mark_time, mark_time + reveal_len)
        else:
            s, e = self._clip_window(cfg.reveal_start, cfg.reveal_start + reveal_len)
        d = max(e - s, 0.0)
        if d <= 0:
            return None
        card_y = cfg.top_bar_h + cfg.card_top_gap
        correct = self.q["correct_answer"]
        if isinstance(correct, (list, tuple, set)):
            correct_letters = [str(c).strip().upper() for c in correct if str(c).strip()]
        elif isinstance(correct, str):
            if "," in correct:
                correct_letters = [c.strip().upper() for c in correct.split(",") if c.strip()]
            else:
                correct_letters = [correct.strip().upper()]
        else:
            correct_letters = [str(correct).strip().upper()]
        correct_letters = [c for c in correct_letters if c]
        hl_x = 90
        hl_w = cfg.width - 180
        hl_h = 90
        reveal_offset_y = 40

        if hasattr(self, "_answer_positions"):
            reveal_offset_y = 0
            first = next((c for c in correct_letters if c in self._answer_positions), None)
            if first is not None:
                y, h = self._answer_positions[first]
                hl_y = y - 18 + reveal_offset_y
                hl_h = h + 36
            else:
                hl_y = card_y + 360 - 18 + reveal_offset_y
        else:
            answer_start_y = card_y + 360
            line_gap = 105
            order = [letter for letter in ["A", "B", "C", "D", "E"] if letter in self.q["choices"]]
            first = correct_letters[0] if correct_letters else (order[0] if order else "A")
            correct_idx = order.index(first) if first in order else 0
            hl_y = answer_start_y + correct_idx * line_gap - 18 + reveal_offset_y

        # green rounded rectangle behind correct answer
        hl_img = self._rounded_rect_rgba((hl_w, hl_h), radius=26, fill_rgba=(0, 255, 127, 55))
        def _pulse_scale(t: float) -> float:
            tt = min(max(t, 0.0), 0.6)
            if tt <= 0.3:
                return 1.0 + 0.07 * (tt / 0.3)
            return 1.07 - 0.07 * ((tt - 0.3) / 0.3)

        highlight = (
            ImageClip(hl_img, ismask=False)
            .set_position((hl_x, hl_y))
            .set_start(s)
            .set_duration(d)
            .fadein(0.3)
            .resize(_pulse_scale)
        )
        # Soft glow behind highlight
        glow_img = self._rounded_rect_rgba((hl_w, hl_h), radius=30, fill_rgba=(0, 255, 127, 35))
        glow = (
            ImageClip(glow_img, ismask=False)
            .set_position((hl_x - 6, hl_y - 6))
            .set_start(s)
            .set_duration(d)
            .fadein(0.4)
        )

        # Subtle shimmer on highlight (opacity pulse via mask)
        def _shimmer_opacity(t: float) -> float:
            tt = min(max(t, 0.0), 0.8)
            return 0.25 + 0.25 * math.sin(2 * math.pi * 1.6 * tt)

        shimmer = ImageClip(hl_img, ismask=False).set_position((hl_x, hl_y)).set_start(s).set_duration(d)
        shimmer_mask = VideoClip(
            make_frame=lambda t: np.ones((hl_h, hl_w)) * _shimmer_opacity(t),
            ismask=True,
            duration=d,
        )
        shimmer = shimmer.set_mask(shimmer_mask)

        layers: List = []

        def add_highlight(y: int, h: int):
            local_h = h + 36
            local_y = y - 18 + reveal_offset_y
            hl_img = self._rounded_rect_rgba((hl_w, local_h), radius=26, fill_rgba=(0, 255, 127, 55))
            highlight = (
                ImageClip(hl_img, ismask=False)
                .set_position((hl_x, local_y))
                .set_start(s)
                .set_duration(d)
                .fadein(0.3)
                .resize(_pulse_scale)
            )
            glow_img = self._rounded_rect_rgba((hl_w, local_h), radius=30, fill_rgba=(0, 255, 127, 35))
            glow = (
                ImageClip(glow_img, ismask=False)
                .set_position((hl_x - 6, local_y - 6))
                .set_start(s)
                .set_duration(d)
                .fadein(0.4)
            )
            shimmer = ImageClip(hl_img, ismask=False).set_position((hl_x, local_y)).set_start(s).set_duration(d)
            shimmer_mask = VideoClip(
                make_frame=lambda t: np.ones((local_h, hl_w)) * _shimmer_opacity(t),
                ismask=True,
                duration=d,
            )
            shimmer = shimmer.set_mask(shimmer_mask)
            layers.extend([glow, highlight, shimmer])

        if hasattr(self, "_answer_positions") and self._answer_positions:
            for letter in correct_letters:
                if letter in self._answer_positions:
                    y, h = self._answer_positions[letter]
                    add_highlight(y, h)
        else:
            layers.extend([glow, highlight, shimmer])

        if hasattr(self, "_answer_positions") and self._answer_positions:
            try:
                fontsize = int(getattr(self, "_answer_fontsize", 44))
            except Exception:
                fontsize = 44
            for letter in correct_letters:
                if letter not in self._answer_positions or letter not in self.q["choices"]:
                    continue
                y, _h = self._answer_positions[letter]
                answer_text = self.make_text_clip(
                    f"{letter}. {self.q['choices'][letter]}",
                    fontsize=fontsize,
                    color_hex=self.cfg.ok,
                    max_width=self.cfg.safe_width,
                    align="left",
                    font_path=self._font(self.cfg.font_bold),
                    shadow=True,
                ).set_position((110, y + reveal_offset_y)).set_start(s).set_duration(d).fadein(0.2)
                answer_text = answer_text.resize(lambda t: 1.0 + 0.06 * min(max(t, 0.0), 0.4) / 0.4)
                layers.append(answer_text)

        self._reveal_actual_end = e
        return CompositeVideoClip(layers, size=(cfg.width, cfg.height), bg_color=None)

    def build_explanation(self):
        cfg = self.cfg
        explain_start = cfg.explain_start
        explain_start = max(explain_start, getattr(self, "_reveal_actual_end", explain_start))
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

    def build_subtitles(self) -> List:
        cfg = self.cfg
        clips: List = []
        subtitle_x = (cfg.width - cfg.safe_width) // 2
        explain_start = max(cfg.explain_start, getattr(self, "_reveal_actual_end", cfg.explain_start))

        marks = self._load_speech_marks()
        if marks:
            for idx, (start, text, _t) in enumerate(marks):
                end = marks[idx + 1][0] if idx + 1 < len(marks) else self.dur
                s, e = self._clip_window(start, end)
                d = max(e - s, 0.0)
                if d <= 0:
                    continue
                fontsize, final_text = self._fit_subtitle_text(text, max_lines=3)
                clip = self.make_text_clip(
                    final_text,
                    fontsize=fontsize,
                    color_hex="#FFFFFF",
                    max_width=cfg.safe_width,
                    align="center",
                    font_path=self._font(cfg.font_med),
                    shadow=True,
                ).set_position((subtitle_x, cfg.height - 260)).set_start(s).set_duration(d)
                clips.append(clip)
            return clips

        def add(txt: str, start: float, end: float):
            s, e = self._clip_window(start, end)
            d = max(e - s, 0.0)
            if d <= 0:
                return
            clip = self.make_text_clip(
                txt,
                fontsize=44,
                color_hex="#FFFFFF",
                max_width=cfg.safe_width,
                align="center",
                font_path=self._font(cfg.font_med),
                shadow=True,
            ).set_position((subtitle_x, cfg.height - 260)).set_start(s).set_duration(d)
            clips.append(clip)

        add("Comment A, B, C, or D.", cfg.engage_start, cfg.engage_end)
        add(f"The correct answer is {self.q['correct_answer']}.", cfg.reveal_start, cfg.reveal_end)
        add(self.explanation, explain_start, self.dur)

        return clips

    def _fit_subtitle_text(self, text: str, max_lines: int = 3) -> Tuple[int, str]:
        # Shrink font to fit max_lines; if still too long, truncate.
        font_path = self._font(self.cfg.font_med)
        min_size = 32
        for fontsize in range(44, min_size - 1, -1):
            try:
                font = ImageFont.truetype(font_path, fontsize)
            except Exception:
                font = ImageFont.load_default()
            lines = self._wrap_lines(text, font, self.cfg.safe_width - 16)
            if len(lines) <= max_lines:
                return fontsize, text

        # Truncate to max_lines at min size.
        try:
            font = ImageFont.truetype(font_path, min_size)
        except Exception:
            font = ImageFont.load_default()
        lines = self._wrap_lines(text, font, self.cfg.safe_width - 16)
        if len(lines) <= max_lines:
            return min_size, text

        trimmed = lines[:max_lines]
        if trimmed:
            last = trimmed[-1]
            if len(last) > 3:
                last = last[:-3].rstrip()
            trimmed[-1] = f"{last}..."
        return min_size, " ".join(trimmed)

    def _load_speech_marks(self) -> List[Tuple[float, str, str]]:
        marks_path = Path(f"{self.audio_path}.marks.json")
        if not marks_path.exists():
            return []
        marks: List[Tuple[float, str, str]] = []
        try:
            with marks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    mark_type = obj.get("type", "")
                    if mark_type != "sentence":
                        continue
                    t = float(obj.get("time", 0.0)) / 1000.0
                    v = str(obj.get("value", "")).strip()
                    if v:
                        marks.append((t, v, mark_type))
        except Exception:
            return []
        return marks

    def _find_sentence_time(self, prefix: str) -> float | None:
        marks = self._load_speech_marks()
        if not marks:
            return None
        target = prefix.strip().lower()
        for t, v, mark_type in marks:
            if mark_type != "sentence":
                continue
            if v.strip().lower().startswith(target):
                return t
        return None

    def build_footer(self):
        cfg = self.cfg
        # Footer branding (always visible, plus it serves as "Outro" messaging)
        footer = self.make_text_clip(
            "@certpulse  •  Follow to pass AWS 🚀",
            fontsize=38,
            color_hex=cfg.accent,
            max_width=self.cfg.width - 120,
            align="center",
            font_path=self._font(self.cfg.font_med),
            shadow=False,
        ).set_position(("center", self.cfg.height - 90)).set_duration(self.dur).set_opacity(0.9)
        return footer

    def build_audio(self):
        music_path = os.getenv("BACKGROUND_MUSIC_PATH", "").strip()
        music_vol = float(os.getenv("BACKGROUND_MUSIC_VOLUME", "0.06"))
        layers = [self.audio]
        if music_path and Path(music_path).exists():
            music = AudioFileClip(music_path).volumex(music_vol).set_duration(self.dur)
            layers.append(music)

        pop_path = Path("assets/sfx/pop.wav")
        if pop_path.exists():
            try:
                mark_time = self._find_sentence_time("the correct answer is")
                pop_start = (mark_time + 0.2) if mark_time is not None else (self.cfg.reveal_start + 0.3)
                pop = AudioFileClip(str(pop_path)).volumex(0.2).set_start(pop_start)
                layers.append(pop)
            except Exception:
                pass

        return CompositeAudioClip(layers) if len(layers) > 1 else self.audio

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

        layers.extend(self.build_countdown())

        engage = self.build_engagement_prompt()
        if engage:
            layers.append(engage)

        reveal = self.build_reveal()
        if reveal:
            layers.append(reveal)

        exp = self.build_explanation()
        if exp:
            layers.append(exp)

        layers.extend(self.build_subtitles())
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
