import io
import base64
from PIL import Image

def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"

def make_img(rgb, size=30):
    """Create a solid RGB swatch of given size."""
    return Image.new("RGB", (size, size), rgb)

def make_b64(rgb, size=30):
    return to_b64(make_img(rgb, size))