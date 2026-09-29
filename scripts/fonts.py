"""Label font for benchmark renders that works without system font packages."""
from functools import lru_cache

from PIL import ImageFont


@lru_cache(maxsize=None)
def ui_font(size):
    # DejaVu is resolved from the system font directories when installed; otherwise
    # Pillow's bundled scalable font keeps rendering deterministic and dependency-free.
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default(size)
