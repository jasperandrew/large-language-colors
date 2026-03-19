import sys

def print_usage():
    print(f"Usage: python {sys.argv[0]} [--debug] [--hue] <N_SAMPLES=int|'test'> <OUT_PATH=str>")

args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]

if len(args) != 2:
    print("Incorrect arguments")
    print_usage()
    exit(1)

test = args[0] == "test"
out_path = args[1] # validate path?

hue_line = True if "-h" in opts or "--hue" in opts else False
debug = True if "-d" in opts or "--debug" in opts else False

if not test:
    try:
        n = int(args[0])
    except:
        print("Invalid N_SAMPLES value: " + args[0])
        print_usage()
        exit(1)

import subprocess
import time
import csv
import os

import tools.swatch as swatch
import tools.llm as llm
import tools.samples as samples
from tools.progress import ProgressBar

def run(cmd):
    subprocess.run(cmd, shell=True)

def icat(img, next_txt=None):
    img.save("tmp.png")
    run("kitten icat -n --align left tmp.png" + (" && echo \r" + next_txt if next_txt != None else ""))
    time.sleep(0.01)
    run("rm tmp.png")


lang_codes = { # ISO 639
    "en": "English",
    "zh": "Chinese (simplified)"
}

lang_code = "en"
prompt = f"You see a solid-colored square. Name its color in {lang_codes[lang_code]}. You may use color names that are as specific or as general as you want. Do not elaborate on or decorate your response, limit it to the name only."
# prompt = f"You see a solid-colored square. Your task is to name its color in {lang_codes[lang_code]}. First, describe the color with a short sentence, ending with a period. Then, give me the hex code of the color. Finally, your response should end with your chosen color name, which may be as specific or as general as you want. I'm trusting you on this, don't let me down."

if not out_path.endswith(".csv"): out_path += ".csv"
file_exists = os.path.exists(out_path)

rgb_data = samples.test_cube() if test else (samples.hue_line(n) if hue_line else samples.full_cube(n))
prog_bar = ProgressBar(len(rgb_data) * len(llm.MODELS), prefix = 'Querying:')

with open(out_path,'a') as f:
    writer = csv.writer(f)
    if not file_exists: writer.writerow(["r","g","b","lang_code"] + llm.MODELS)
    
    for rgb in rgb_data:
        img = swatch.make_img(rgb)
        msgs = []
        for model in llm.MODELS:
            resp = llm.query(model, img, prompt).replace("\n", " ")
            msgs.append(resp)
            prog_bar.iterate()
        r,g,b = rgb
        writer.writerow([r,g,b,lang_code] + msgs)
        f.flush()

        if debug:
            print(f"\nrgb: {tuple(map(int, rgb))}")
            icat(img, "responses:")
            for i,model in enumerate(llm.MODELS):
                if len(msgs): print(f"{model}: {msgs[i]}")
                else:         print("test")