import sys

def err_exit(err_msg=None):
    if err_msg: print(f"[ERROR] {err_msg}")
    print(f"Usage: python {sys.argv[0]} <LANG_CODE> <'full'|'hue'> <COUNT:int> <OUT_DIR> [--name=FILE_NAME] [--print]")
    print(f"       python {sys.argv[0]} <LANG_CODE> test-cube <OUT_DIR> [--name=FILE_NAME] [--print]")
    print(f"Notes:")
    print(f"       If the output file already exists, data will be appended.")
    exit(1)

args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]

if "--help" in opts: err_exit()

print_results = False
file_name = None
for opt in opts:
    if opt == "-p" or opt == "--print": print_results = True
    if opt.startswith("-n") or opt.startswith("--name"):
        vals = opt.split("=")
        if len(vals) != 2: err_exit("Invalid arguments")
        file_name = vals[1]
        if not file_name.endswith(".csv"): err_exit("File name must end with '.csv'")

rgb_range = None
count = None
test = False

if len(args) == 4:
    lang_code, rgb_range, count, out_dir = args
    if rgb_range not in ["full","hue"]: err_exit("Invalid arguments")
    try: count = int(count)
    except: err_exit("Invalid COUNT: " + count)
elif len(args) == 3:
    lang_code, test, out_dir = args
    test = (test == "test-cube")
    if not test: err_exit(f"Invalid arguments")
else:
    err_exit("Invalid arguments")

if not out_dir.endswith("/"): out_dir += "/"
out_path = out_dir + (file_name if file_name else f"{lang_code}-{"test-cube" if test else f"{rgb_range}-raw"}.csv")

lang_prompts = { # ISO 639
    "en": "You see a solid-colored square. Name its color in English. You may use color names that are as specific or as general as you want. Do not elaborate on or decorate your response, limit it to the name only.",
    # en: "You see a solid-colored square. Your task is to name its color in English. First, describe the color with a short sentence, ending with a period. Then, give me the hex code of the color. Finally, your response should end with your chosen color name, which may be as specific or as general as you want. I'm trusting you on this, don't let me down."
    "zh": "看到一个纯色的正方形。用中文说出它的颜色。可以使用任意具体或笼统的颜色名称。不要对回答进行任何修饰或扩展，只需说出颜色名称。"
}

if lang_code not in lang_prompts.keys(): err_exit(f"Unsupported LANG_CODE: {lang_code}")
prompt = lang_prompts[lang_code]

import subprocess
import time
import csv
import os

import tools.swatch as swatch
import tools.llm as llm
import tools.samples as samples
from tools.progress import ProgressBar

def icat(img, next_txt=None):
    def run(cmd):
        subprocess.run(cmd, shell=True)
    img.save("tmp.png")
    run("kitten icat -n --align left tmp.png" + (" && echo \r" + next_txt if next_txt != None else ""))
    time.sleep(0.01)
    run("rm tmp.png")

file_exists = os.path.exists(out_path)

rgb_data = samples.test_cube() if test else (samples.hue_line(count) if rgb_range == "hue" else samples.full_cube(count))
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

        if print_results:
            print(f"\nrgb: {tuple(map(int, rgb))}")
            icat(img, "responses:")
            for i,model in enumerate(llm.MODELS):
                if len(msgs): print(f"{model}: {msgs[i]}")
                else:         print("test")