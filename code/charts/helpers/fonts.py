import os
import matplotlib as mpl
from matplotlib import font_manager

def needs_cjk(texts: list[str]) -> bool:
    """Return True if any string in texts contains CJK or Hangul characters."""
    for text in texts:
        for ch in text:
            cp = ord(ch)
            if (
                0x1100 <= cp <= 0x11FF    # Hangul Jamo
                or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables
                or 0x3040 <= cp <= 0x30FF  # Hiragana / Katakana
                or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
                or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
                or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
                or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
            ):
                return True
    return False


_CJK_FONT_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK KR", "Noto Sans CJK TC", "Noto Sans CJK JP",
    "PingFang SC", "Apple SD Gothic Neo", "Hiragino Sans",
    "Microsoft YaHei", "Malgun Gothic", "MS Gothic",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "UnDotum", "NanumGothic",
    "SimHei", "SimSun", "NSimSun",
]

_CJK_KEYWORDS = ("cjk", "noto", "gothic", "hei", "yuan", "ming", "dotum",
                  "gulim", "batang", "nanum", "malgun", "pingfang", "hiragino")


@lru_cache(maxsize=1)
def _find_cjk_font() -> Optional[str]:
    """
    Search the system font registry for a CJK-capable family.
    Result is cached — font manager scans can be slow.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available:
            return candidate
    for font in font_manager.fontManager.ttflist:
        if any(kw in font.name.lower() for kw in _CJK_KEYWORDS):
            return font.name
    return None


def configure_font(font_spec: Optional[str], labels: Optional[list[str]] = None) -> None:
    """
    Set matplotlib's active font.

    - font file path  → register and activate it
    - font family name → activate directly
    - None            → auto-detect a CJK font if labels require one
    """
    if font_spec is None:
        if labels and needs_cjk(labels):
            font_spec = _find_cjk_font()
            if font_spec:
                print(f"Auto-detected CJK font: '{font_spec}'")
            else:
                print(
                    "Warning: CJK/Hangul characters found in labels but no CJK font "
                    "is installed. Install a Noto CJK font for correct rendering."
                )
                return
        else:
            return

    try:
        if os.path.exists(font_spec):
            font_manager.fontManager.addfont(font_spec)
            fp = font_manager.FontProperties(fname=font_spec)
            font_spec = fp.get_name()
            print(f"Loaded font from file: '{font_spec}'")

        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [font_spec] + list(
            mpl.rcParams.get("font.sans-serif", [])
        )
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception as exc:
        print(f"Warning: could not set font '{font_spec}': {exc}")
