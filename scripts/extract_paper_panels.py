"""Cut the 12 accuracy-vs-time panels of the FedASMU paper appendix (arXiv 2312.05770v1,
Figures 3-6, pages 17-18) out of the PDF and record each panel's exact axes.

The paper is licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); the
panels are reproduced unchanged except for cropping, with attribution, for side-by-side
comparison in scripts/appendix_figures.py.

How the axes are recovered (no manual reading of numbers):
  1. pdftoppm renders pages 17-18 at 300 dpi; the axes frame of each panel is found as
     a pair of long horizontal black lines with the same x extent.
  2. pdftotext -bbox gives the position of every tick label (the figures are vector
     graphics, so the labels are real text).  A straight-line fit pixel -> value over
     the tick labels gives the data value at the frame edges, i.e. xlim / ylim.
     The fit residual over the ticks is ~1e-9, so the mapping is exact.

Output: results/figures/paper_appendix/{<model>-<dataset>.png, axes.json}
Requires poppler-utils (pdftoppm, pdftotext).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "figures", "paper_appendix")
URL = "https://arxiv.org/pdf/2312.05770v1"
MD5 = "f1278f88ee70fb7676896fa9655d38ab"
DPI = 300
PANELS = {17: [["lenet5-cifar10", "cnn-cifar10", "resnet20-tinyimagenet"],      # Figure 3
               ["lenet5-cifar100", "cnn-cifar100", "resnet20-cifar100"],        # Figure 4
               ["alexnet-cifar10", "vgg11-cifar10", "textcnn-imdb"]],           # Figure 5
          18: [["alexnet-cifar100", "vgg11-cifar100", "lenet5-fmnist"]]}        # Figure 6


def hlines(img):
    a = np.asarray(img.convert("L")) < 80
    out = []
    for y in range(a.shape[0]):
        d = np.diff(np.concatenate([[0], a[y].astype(int), [0]]))
        for x0, x1 in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if x1 - x0 > 400:
                out.append((y, int(x0), int(x1)))
    return out


def frames(img, top_only):
    L = hlines(img)
    fr = []
    for ya, a0, a1 in L:
        for yb, b0, b1 in L:
            if 330 < yb - ya < 430 and abs(a0 - b0) <= 3 and abs(a1 - b1) <= 3 and a1 - a0 > 500:
                f = (a0, ya, a1, yb)
                if not any(abs(f[0] - g[0]) < 20 and abs(f[1] - g[1]) < 30 for g in fr):
                    fr.append(f)
    if top_only:                      # page 18 also holds Figure 7 below Figure 6
        fr = [f for f in fr if f[1] < 900]
    return sorted(fr, key=lambda f: (round(f[1] / 200), f[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="local copy of arXiv 2312.05770v1 (downloaded if omitted)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp()
    pdf = a.pdf or os.path.join(tmp, "fedasmu.pdf")
    if not a.pdf:
        urllib.request.urlretrieve(URL, pdf)
    md5 = hashlib.md5(open(pdf, "rb").read()).hexdigest()
    if md5 != MD5:
        print(f"warning: PDF md5 {md5} differs from the v1 file used here ({MD5})")
    subprocess.run(["pdftoppm", "-r", str(DPI), "-f", "17", "-l", "18", "-png", pdf, os.path.join(tmp, "pg")], check=True)
    subprocess.run(["pdftotext", "-bbox", "-f", "17", "-l", "18", pdf, os.path.join(tmp, "bbox.html")], check=True)
    txt = open(os.path.join(tmp, "bbox.html")).read()
    s = DPI / 72
    words = []
    for i, p in enumerate(re.findall(r"<page[^>]*>(.*?)</page>", txt, re.S)):
        for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', p):
            if re.fullmatch(r"-?\d+(\.\d+)?", m.group(5)):
                words.append((17 + i, *(float(v) * s for v in m.groups()[:4]), float(m.group(5))))
    res = {}
    for pg, rows in PANELS.items():
        img = Image.open(os.path.join(tmp, f"pg-{pg}.png"))
        fr = frames(img, pg == 18)
        assert len(fr) == 3 * len(rows), (pg, fr)
        for k, (x0, y0, x1, y1) in enumerate(fr):
            name = rows[k // 3][k % 3]
            W = [w for w in words if w[0] == pg]
            yt = [(w[5], (w[2] + w[4]) / 2) for w in W if x0 - 80 < w[3] < x0 + 2 and y0 - 15 < (w[2] + w[4]) / 2 < y1 + 15]
            xt = [(w[5], (w[1] + w[3]) / 2) for w in W if y1 - 15 < w[2] < y1 + 40 and x0 - 20 < (w[1] + w[3]) / 2 < x1 + 20]
            fy = np.polyfit([p for _, p in yt], [v for v, _ in yt], 1)
            fx = np.polyfit([p for _, p in xt], [v for v, _ in xt], 1)
            resid = max(max(abs(np.polyval(fy, p) - v) for v, p in yt), max(abs(np.polyval(fx, p) - v) / 1e5 for v, p in xt))
            # frame lines are ~2 px thick; use their centre
            ylim = [float(np.polyval(fy, y1 + 1)), float(np.polyval(fy, y0 + 1))]
            xlim = [float(np.polyval(fx, x0 + 1)), float(np.polyval(fx, x1 - 1))]
            img.crop((x0 - 92, y0 - 12, x1 + 14, y1 + 62)).save(os.path.join(OUT, f"{name}.png"))
            res[name] = dict(source=f"arXiv 2312.05770v1 page {pg}", frame_px_300dpi=[x0, y0, x1, y1],
                             xlim=xlim, ylim=ylim, xticks=sorted(v for v, _ in xt), yticks=sorted(v for v, _ in yt),
                             fit_residual=float(resid))
            print(f"{name:24s} xlim=({xlim[0]:.0f}, {xlim[1]:.0f}) ylim=({ylim[0]:.4f}, {ylim[1]:.4f}) resid={resid:.1e}")
    json.dump(res, open(os.path.join(OUT, "axes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
