import tools.local as keys # local.py contains API keys
import tools.swatch as swatch

# OPENAI CHATGPT

import openai

CHATGPT_MODEL = "gpt-5.4-mini-2026-03-17"
CHATGPT = openai.OpenAI(api_key=keys.OPENAI_API_KEY)

def query_chatgpt(img, prompt):
    input = [{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": swatch.to_b64(img)}, # for chatgpt, pass the image as a b64 uri
            {"type": "input_text", "text": prompt},
        ],
    }]
    resp = CHATGPT.responses.create(model=CHATGPT_MODEL, input=input)
    for r in resp.output:
        if r.type == "message": return r.content[0].text

    raise ValueError('ChatGPT Error: No response message found.')


# GOOGLE GEMINI

import google.generativeai as genai

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI = genai.GenerativeModel(GEMINI_MODEL)
genai.configure(api_key=keys.GOOGLE_API_KEY)

def query_gemini(img, prompt):
    input = [img, prompt] # for gemini, just pass the image object
    return GEMINI.generate_content(input).text


# GENERAL FUNCTIONALITY

MODELS = ["chatgpt", "gemini"]

def query(model, img, prompt):
    if model == "chatgpt":
        return query_chatgpt(img, prompt)
    if model == "gemini":
        return query_gemini(img, prompt)
    
    raise ValueError(f'Invalid model string ({model}).')
