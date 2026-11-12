"""Read the data points of the curves in the FedASMU paper appendix Figures 3-6 directly
from the PDF's vector graphics (arXiv 2312.05770v1, CC BY 4.0).

The figures are matplotlib vector plots.  pdftocairo -svg turns pages 17-18 into SVG; every
curve is one stroked <path> with the method's colour and a transform to page points.  The
page points are mapped to data values with the axes measured by
scripts/extract_paper_panels.py (results/figures/paper_appendix/axes.json), so no value is
read by eye.  Check: the last point of each digitised curve is compared with the final
accuracy the paper reports in Tables 1-2 (printed by this script).

Output: results/figures/paper_appendix/curves.csv  (workload, method, time_s, accuracy)
Requires poppler-utils (pdftocairo).
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PA = os.path.join(ROOT, "results", "figures", "paper_appendix")
URL = "https://arxiv.org/pdf/2312.05770v1"
PX = 300 / 72  # axes.json frames are in 300-dpi pixels; SVG coordinates are points
# legend order of the paper (rainbow colour map) -> stroke colour in the SVG
COLORS = {
    "FedASMU": "100%,0%,0%",
    "FedAvg": "100%,33.815002%,17.16156%",
    "FedProx": "100%,63.645935%,33.815002%",
    "MOON": "83.332825%,86.601257%,50%",
    "FedDyn": "61.372375%,98.408508%,64.120483%",
    "FedAsync": "38.626099%,98.408508%,76.734924%",
    "PORT": "16.665649%,86.601257%,86.601257%",
    "ASO-Fed": "6.077576%,63.645935%,94.108582%",
    "FedBuff": "28.038025%,33.815002%,98.51532%",
    "FedSA": "50%,0%,100%",
}


def paths(svg):
    out = []
    for attrs in re.findall(r"<path ([^>]*)/>", svg):
        m = re.search(r"stroke:rgb\(([^)]*)\)", attrs)
        d = re.search(r' d="([^"]*)"', attrs)
        tr = re.search(r'transform="matrix\(([^)]*)\)"', attrs)
        if not (m and d and tr):
            continue
        nums = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", d.group(1))]
        if len(nums) < 40:  # markers and legend lines are short
            continue
        a, b, c, dd, e, f = (float(v) for v in tr.group(1).split(","))
        xy = np.array(nums).reshape(-1, 2)
        X = a * xy[:, 0] + c * xy[:, 1] + e
        Y = b * xy[:, 0] + dd * xy[:, 1] + f
        out.append((m.group(1), X * PX, Y * PX))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="local copy of arXiv 2312.05770v1 (downloaded if omitted)")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    pdf = a.pdf or os.path.join(tmp, "fedasmu.pdf")
    if not a.pdf:
        urllib.request.urlretrieve(URL, pdf)
    axes = json.load(open(os.path.join(PA, "axes.json")))
    svgs = {}
    for pg in (17, 18):
        f = os.path.join(tmp, f"p{pg}.svg")
        subprocess.run(["pdftocairo", "-svg", "-f", str(pg), "-l", str(pg), pdf, f], check=True)
        svgs[pg] = paths(open(f).read())
    rows = []
    for wl, ax in axes.items():
        pg = int(ax["source"].split()[-1])
        x0, y0, x1, y1 = ax["frame_px_300dpi"]
        for meth, col in COLORS.items():
            best = None
            for c, X, Y in svgs[pg]:
                if c != col:
                    continue
                inside = (X > x0 - 2) & (X < x1 + 2) & (Y > y0 - 2) & (Y < y1 + 2)
                if inside.mean() > 0.95 and (best is None or len(X) > len(best[0])):
                    best = (X, Y)
            if best is None:
                continue
            X, Y = best
            t = ax["xlim"][0] + (X - (x0 + 1)) / ((x1 - 1) - (x0 + 1)) * (ax["xlim"][1] - ax["xlim"][0])
            acc = ax["ylim"][0] + ((y1 + 1) - Y) / ((y1 + 1) - (y0 + 1)) * (ax["ylim"][1] - ax["ylim"][0])
            rows += [(wl, meth, float(u), float(v)) for u, v in zip(t, acc)]
    df = pd.DataFrame(rows, columns=["workload", "method", "time_s", "accuracy"])
    df.to_csv(os.path.join(PA, "curves.csv"), index=False)
    last = df.groupby(["workload", "method"]).tail(1).pivot(index="workload", columns="method", values="accuracy")
    print("last digitised point per curve (compare with paper Tables 1-2):")
    print(last[["FedASMU", "FedBuff", "FedAvg"]].round(3).to_string())
    print(df.groupby(["workload", "method"]).size().unstack().fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
