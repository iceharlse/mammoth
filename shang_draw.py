import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from pathlib import Path
from matplotlib import font_manager

matplotlib.set_loglevel("error")

CSV_PATH = "./defect_log_20251222-152809.csv"
OUT_PNG = "./entropy_only_axes_TNR_big.png"
OUT_PDF = "./entropy_only_axes_TNR_big.pdf"

assert Path(CSV_PATH).exists(), f"CSV not found: {CSV_PATH}"

# --- Font + sizes ---
FS_BASE = 22
FS_LABEL = 26
FS_TICK = 22

mpl.rcParams.update({
    # Times New Roman (fallback to Times/serif if missing)
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    # embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,

    # sizes
    "font.size": FS_BASE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
})

# Optional: check if Times New Roman is found (won't stop plotting if not)
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
tnr_available = "Times New Roman" in available_fonts

df = pd.read_csv(CSV_PATH)
q10 = df["min_entropy"].quantile(0.10)

# two-band jitter
rng = np.random.default_rng(0)
y = df["selected_is_correct"].to_numpy(float)
noise = rng.normal(0, 1, size=len(df))

y_jit = y.copy()
top = y == 1
bot = ~top
y_jit[top] = np.clip(1.0 + noise[top] * 0.03, 0.82, 1.10)
y_jit[bot] = np.clip(0.0 + noise[bot] * 0.03, -0.10, 0.18)

reg_mask = df["regret"].to_numpy() == 1

fig, ax = plt.subplots(figsize=(7.2, 7.2))

ax.scatter(df["min_entropy"], y_jit, s=10, alpha=0.25, linewidths=0)
ax.scatter(df.loc[reg_mask, "min_entropy"], y_jit[reg_mask],
           s=55, marker="x", linewidths=1.8)

ax.axvline(q10, linestyle="--")

ax.set_xscale("log")
ax.set_ylim(-0.1, 1.15)
ax.set_xlim(df["min_entropy"].min() * 0.8, 0.3)

ax.set_title("")
ax.set_xlabel("Minimum predictive entropy (log)")
ax.set_ylabel("Correctness (0/1)")

# ticks style
ax.tick_params(axis="both", which="major", length=7, width=1.4)
ax.tick_params(axis="both", which="minor", length=4, width=1.1)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.show()

{"png": OUT_PNG, "pdf": OUT_PDF, "times_new_roman_available": tnr_available}
