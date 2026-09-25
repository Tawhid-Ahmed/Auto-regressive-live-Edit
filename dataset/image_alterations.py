"""On-the-fly PIL transforms for Step 3 image-alteration probes.

Applied at eval load time to the *same* entity image used during editing.
Edit always uses the original; generality probes use altered versions.
"""
from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter

ALTERATION_NAMES = (
    "low_res",
    "rotate_90",
    "mirror",
    "crop_center",
    "mild",
)

ALTERATION_LABELS = {
    "low_res": "Low resolution",
    "rotate_90": "Rotate 90 deg",
    "mirror": "Mirror (H-flip)",
    "crop_center": "Center crop",
    "mild": "Mild blur/brightness",
}


class AlteredImageRef:
    """Lazy image handle: load from disk and apply alteration at eval time."""

    __slots__ = ("path", "alteration")

    def __init__(self, path: str, alteration: str | None = None) -> None:
        self.path = path
        self.alteration = alteration

    def __repr__(self) -> str:
        if self.alteration:
            return "AlteredImageRef(%r, %r)" % (self.path, self.alteration)
        return "AlteredImageRef(%r)" % self.path


def apply_alteration(img: Image.Image, name: str) -> Image.Image:
    """Return a copy of ``img`` with the named alteration."""
    if name not in ALTERATION_NAMES:
        raise ValueError("Unknown alteration %r (choose from %s)" % (name, ALTERATION_NAMES))

    w, h = img.size

    if name == "low_res":
        short = min(w, h)
        scale = 112.0 / short
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        small = img.resize((nw, nh), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    if name == "rotate_90":
        return img.transpose(Image.ROTATE_270)

    if name == "mirror":
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    if name == "crop_center":
        cw = max(1, int(round(w * 0.75)))
        ch = max(1, int(round(h * 0.75)))
        left = (w - cw) // 2
        top = (h - ch) // 2
        cropped = img.crop((left, top, left + cw, top + ch))
        return cropped.resize((w, h), Image.BILINEAR)

    if name == "mild":
        out = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        out = ImageEnhance.Brightness(out).enhance(0.92)
        out = ImageEnhance.Contrast(out).enhance(0.95)
        return out

    raise ValueError(name)
