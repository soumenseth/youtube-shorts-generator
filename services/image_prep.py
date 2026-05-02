"""Shared PIL image-preparation helpers used by VideoAssemblyService and SlideshowSceneRenderer."""
from PIL import Image, ImageDraw, ImageFont


def resize_crop(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    tw, th = target
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    return img.crop((left, top, left + tw, top + th))


def add_overlay(img: Image.Image, text: str) -> Image.Image:
    w, h = img.size
    font_size = max(36, h // 22)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    words = text.split()
    lines, line = [], []
    draw_tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for word in words:
        test_line = " ".join(line + [word])
        bbox = draw_tmp.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > w - 80 and line:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))

    line_height = draw_tmp.textbbox((0, 0), "Ag", font=font)[3] + 8
    block_h = line_height * len(lines) + 32
    y_start = h - block_h - 60

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    pad = 24
    od.rectangle([pad, y_start - pad, w - pad, y_start + block_h], fill=(0, 0, 0, 180))
    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)

    final = img_rgba.convert("RGB")
    draw = ImageDraw.Draw(final)
    for i, ln in enumerate(lines):
        bbox = draw.textbbox((0, 0), ln, font=font)
        x = (w - (bbox[2] - bbox[0])) // 2
        y = y_start + i * line_height
        draw.text((x, y), ln, font=font, fill=(255, 255, 255))

    return final
