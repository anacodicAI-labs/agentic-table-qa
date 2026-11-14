"""fig4: accuracy by task family, three configurations, two domains.

Biology uses the MATCHED subset (the agent's 150 questions) so all three
configurations are scored on identical items. CompSci is the full single run.
Answered items only, so agent turn-budget exhaustion is not double-counted here; it is
reported separately in the paper.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical slots 1-3, fixed order, from the reference palette (light surface)
COLORS = {"baseline": "#2a78d6", "cot": "#eb6834", "agent": "#1baf7a"}
HATCH = {"baseline": "", "cot": "///", "agent": "..."}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"

DATA = {
    "Biology (matched, $n=150$)": {
        "baseline": [76.5, 32.6], "cot": [75.0, 40.7], "agent": [58.2, 38.2]},
    "CompSci ($n=608$)": {
        "baseline": [66.9, 40.2], "cot": [66.9, 49.4], "agent": [59.0, 44.3]},
}
FAMILIES = ["Lookup", "Compute"]

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
fig.patch.set_facecolor(SURFACE)
x = np.arange(len(FAMILIES))
w = 0.26

for ax, (title, series) in zip(axes, DATA.items()):
    ax.set_facecolor(SURFACE)
    for i, (name, vals) in enumerate(series.items()):
        off = (i - 1) * (w + 0.015)        # 2px-equivalent gap between fills
        bars = ax.bar(x + off, vals, w, label=name, color=COLORS[name],
                      hatch=HATCH[name], edgecolor=SURFACE, linewidth=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK, pad=8)
    ax.set_xticks(x); ax.set_xticklabels(FAMILIES, fontsize=9.5, color=INK)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED); ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0, labelsize=8.5, colors=MUTED)

axes[0].set_ylabel("accuracy on answered items (%)", fontsize=9, color=INK)
axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
               bbox_to_anchor=(1.03, -0.14))
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig("figures/fig4_accuracy_by_family.pdf", bbox_inches="tight",
            facecolor=SURFACE)
print("wrote figures/fig4_accuracy_by_family.pdf")
