# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/bria_fibo2/text_images.py
# images of the text a caption asks for, drawn the way Fibo 2.0's training drew them: each text in one line, black on
# white, in Pillow's built-in font at size 20. a character it lacks is drawn with the first fallback font that has it:
# training used Aileron, then DejaVu Sans (in the fonts folder of briaai/Fibo-2.0-dev).

import functools
from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

FONT_SIZE = 20
MAX_LINE_CHARS = 256  # longer texts are split into lines of this many characters
PADDING = 2
LINE_SPACING = 4


@functools.cache
def load_font(path: str) -> ImageFont.FreeTypeFont:
    # basic layout like the built-in font, so the text doesn't set differently where libraqm is installed
    return ImageFont.truetype(path, FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)


@functools.cache
def _default_font() -> ImageFont.FreeTypeFont:
    font = ImageFont.load_default(FONT_SIZE)
    assert isinstance(font, ImageFont.FreeTypeFont), "needs Pillow built with FreeType"
    return font


@functools.lru_cache(maxsize=4096)
def _has_glyph(font: ImageFont.FreeTypeFont, char: str) -> bool:
    # a font without a glyph draws its "notdef" box, as it does for U+FFFE, which no font has
    try:
        notdef = font.getmask("\ufffe")
        mask = font.getmask(char)
    except Exception:  # noqa: BLE001 - as in training, a font that can't say is taken to have the glyph
        return True
    return not (mask.size == notdef.size and bytes(mask) == bytes(notdef))


def draw_text_image(text: str, fallback_fonts: Sequence[ImageFont.FreeTypeFont] = ()) -> Image.Image:
    font = _default_font()
    line = " ".join(text.split())
    lines = [line[i : i + MAX_LINE_CHARS] for i in range(0, len(line), MAX_LINE_CHARS)] or [""]
    if all(ord(c) < 128 or _has_glyph(font, c) for c in line):
        return _draw_one_font(lines, font)
    return _draw_mixed(lines, [font, *fallback_fonts])


# each line is placed by its bounding box, and is at least as tall as "Ag"
def _draw_one_font(lines: list[str], font: ImageFont.FreeTypeFont) -> Image.Image:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    min_box = draw.textbbox((0, 0), "Ag", font=font)
    min_line_height = max(1, min_box[3] - min_box[1])
    boxes = [draw.textbbox((0, 0), line or " ", font=font) for line in lines]
    heights = [max(min_line_height, box[3] - box[1]) for box in boxes]
    width = max([1] + [box[2] - box[0] for box in boxes])
    height = sum(heights) + LINE_SPACING * (len(lines) - 1)

    image = Image.new("RGB", (width + PADDING * 2, height + PADDING * 2), "white")
    draw = ImageDraw.Draw(image)
    y = PADDING
    for line, box, line_height in zip(lines, boxes, heights):
        draw.text((PADDING - box[0], y - box[1]), line, font=font, fill=(0, 0, 0))
        y += line_height + LINE_SPACING
    return image


# runs of characters drawn with the same font: the first font with the glyph, or the built-in font if none has it
def _font_runs(text: str, fonts: Sequence[ImageFont.FreeTypeFont]) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    runs = []
    for c in text:
        font = fonts[0]
        if not (ord(c) < 128 or _has_glyph(fonts[0], c)):
            font = next((f for f in fonts[1:] if _has_glyph(f, c)), fonts[0])
        if runs and runs[-1][1] is font:
            runs[-1][0] += c
        else:
            runs.append([c, font])
    return [(run, font) for run, font in runs]


# each run is drawn with its own font on the line's baseline. lines are as tall as the built-in font's ascent + descent
def _draw_mixed(lines: list[str], fonts: Sequence[ImageFont.FreeTypeFont]) -> Image.Image:
    ascent, descent = fonts[0].getmetrics()
    line_height = max(1, ascent + descent)
    runs_per_line = [_font_runs(line or " ", fonts) for line in lines]
    width = max(max(1, round(sum(font.getlength(run) for run, font in runs))) for runs in runs_per_line)
    height = line_height * len(lines) + LINE_SPACING * (len(lines) - 1)

    image = Image.new("RGB", (width + PADDING * 2, height + PADDING * 2), "white")
    draw = ImageDraw.Draw(image)
    y = PADDING
    for runs in runs_per_line:
        x = float(PADDING)
        for run, font in runs:
            draw.text((x, y + ascent), run, font=font, fill=(0, 0, 0), anchor="ls")
            x += font.getlength(run)
        y += line_height + LINE_SPACING
    return image
