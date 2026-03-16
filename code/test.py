import subprocess
import colorsys
import time
import csv
import numpy as np

from progress import ProgressBar
import swatch
import llm

def run(cmd):
    subprocess.run(cmd, shell=True)

def icat(img, next_txt=None):
    img.save("tmp.png")
    run("kitten icat -n --align left tmp.png" + (" && echo \r" + next_txt if next_txt != None else ""))
    time.sleep(0.01)
    run("rm tmp.png")

def hue_to_rgb(h_idx, n_bins=36, s=1.0, v=1.0):
    h = (h_idx / n_bins)  # in [0,1)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

lang = "English"
prompt = f"You see a solid-colored square. Name its color in {lang}. You may use color names that are as specific or as general as you want. Do not elaborate on or decorate your response, limit it to the name only."

n_samples = 100
samples = list(map(tuple, np.random.randint(0, 256, size=(n_samples,3), dtype=np.uint8)))

prog_bar = ProgressBar(len(samples) * len(llm.MODELS), prefix = 'Querying:')

results = []
for rgb in samples:
    img = swatch.make_img(rgb)
    msgs = []
    for model in llm.MODELS:
        msgs.append(llm.query(model, img, prompt))
        prog_bar.iterate()
    results.append((rgb, img, msgs))

with open('results.csv','a') as f:
    writer = csv.writer(f)

    for res in results:
        rgb,img,msgs = res
        r,g,b = rgb
        writer.writerow([r,g,b,lang.lower()] + msgs)

        print(f"\nrgb: {rgb}")
        icat(img, "responses:")
        for i,model in enumerate(llm.MODELS):
            if len(msgs):
                print(f"{model}: {msgs[i]}")
            else:
                print("test")