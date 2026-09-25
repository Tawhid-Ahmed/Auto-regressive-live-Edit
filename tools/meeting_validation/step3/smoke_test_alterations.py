"""Verify alteration transforms change pixel content."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from PIL import Image
from dataset.image_alterations import apply_alteration, ALTERATION_NAMES

img = Image.new("RGB", (400, 300), color=(120, 80, 200))
for name in ALTERATION_NAMES:
    out = apply_alteration(img, name)
    assert out.size == img.size, name
    if name != "mild":
        assert out.tobytes() != img.tobytes(), "alteration %s had no effect" % name
print("OK - all alterations change image content")
