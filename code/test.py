import io
import base64
import colorsys
from PIL import Image

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"

def make_swatch_img(rgb, size=30):
    """Create a solid RGB swatch of given size."""
    return Image.new("RGB", (size, size), rgb)

def make_swatch_b64(rgb, size=30):
    return img_to_b64(make_swatch_img(rgb, size))

import openai
import google.generativeai as genai
import local

CHATGPT = openai.OpenAI(api_key=local.OPENAI_API_KEY)
CHATGPT_MODEL = "gpt-5-nano"
def query_chatgpt(img, prompt):
    input = [{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": img_to_b64(img)}, # for chatgpt, pass the image as a b64 uri
            {"type": "input_text", "text": prompt},
        ],
    }]
    resp = CHATGPT.responses.create(model=CHATGPT_MODEL, input=input)
    for r in resp.output:
        if r.type == "message": return r.content[0].text
    return "<<ERR: No response message found>>"

genai.configure(api_key=local.GOOGLE_API_KEY)
GEMINI_MODEL = genai.GenerativeModel('gemini-2.5-flash')
def query_gemini(img, prompt):
    input = [img, prompt] # for gemini, just pass the image object
    return GEMINI_MODEL.generate_content(input).text

def query_rgb(model, rgb, prompt):
    if model == "chatgpt":
        return query_chatgpt(make_swatch_img(rgb), prompt)
    if model == "gemini":
        return query_gemini(make_swatch_img(rgb), prompt)
    return f"<<ERR: Unknown model ({model})>>"

basic_color_prompt = "You see a solid-colored square. Name its basic color category in English. Answer with only a single word."

print(query_rgb("chatgpt", (255,0,0), basic_color_prompt))
print(query_rgb("gemini", (255,0,0), basic_color_prompt))