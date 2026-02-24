import swatch
import llm
import subprocess
import colorsys
import time
from progress import ProgressBar

def icat(img):
    img.save("tmp.png")
    subprocess.run("kitten icat --align left tmp.png".split())
    time.sleep(0.01)
    subprocess.run("rm tmp.png".split())

def hue_to_rgb(h_idx, n_bins=36, s=1.0, v=1.0):
    h = (h_idx / n_bins)  # in [0,1)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))

prompt = "You see a solid-colored square. Name its basic color category in English. Answer with only a single word."
samples = []
for idx in [0, 6, 12, 18, 24, 30]:
    rgb = hue_to_rgb(idx, n_bins=36, s=1.0, v=1.0)
    samples.append(rgb)

progress = ProgressBar(len(samples) * len(llm.MODELS), prefix = 'Querying:')

results = []
for rgb in samples:
    img = swatch.make_img(rgb)
    msgs = []
    for model in llm.MODELS:
        msgs.append(llm.query(model, img, prompt))
        progress.iterate()
    results.append((rgb, img, msgs))

print()
for r in results:
    rgb,img,msgs = r
    print(f"rgb: {rgb}")
    for i,model in enumerate(llm.MODELS):
        print(f"{model}: {msgs[i]}")
    icat(img)