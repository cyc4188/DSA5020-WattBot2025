"""Generate publication-grade vector PDF figures for the DSA5020 ACM report."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 8.8,
    "axes.labelsize": 8.3,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.45,
})

C_NAVY = "#1B365D"
C_BLUE = "#0072B2"
C_TEAL = "#009E73"
C_ORANGE = "#D55E00"
C_PURPLE = "#7B3294"
C_GRAY = "#5C6670"


def make_fig1_architecture():
    """Clean horizontal two-column architecture diagram of the Pure-RAG Cascade."""
    fig, ax = plt.subplots(figsize=(7.0, 2.30))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#FFFFFF", ec=C_NAVY, lw=1.0, title_c=C_NAVY, fontsize=7.0):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top",
                fontsize=fontsize + 0.3, fontweight="bold", color=title_c, zorder=4)
        if body:
            ax.text(x + w / 2, y + (h - 3.6) / 2, body, ha="center", va="center",
                    fontsize=fontsize - 0.35, color="#222222", linespacing=1.22, zorder=4)

    def arrow(x1, y1, x2, y2, label="", color="#333333", rad=0.0, lx=0, ly=0, ls="-"):
        a = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=4.4,head_width=2.5",
            color=color, linewidth=1.0, linestyle=ls, zorder=5
        )
        ax.add_patch(a)
        if label:
            mx, my = (x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly
            ax.text(mx, my, label, ha="center", va="center", fontsize=6.4,
                    color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.92), zorder=6)

    bands = [
        (0.8, 1.0, 20.5, 35.8, "#F5F7FA", "#C8D2DC", "Offline Corpus Layer"),
        (23.3, 1.0, 34.0, 35.8, "#F2F8FC", "#B8D4E8", "Tier-1 Parallel Pure-RAG Solvers"),
        (59.3, 1.0, 24.0, 35.8, "#FDF7F0", "#E8C8A8", "Difficulty-Aware Gate & Tier-2"),
        (85.0, 1.0, 14.2, 35.8, "#F1F9F5", "#B0D9C5", "Output Layer"),
    ]
    for x, y, w, h, fc, ec, lbl in bands:
        bg = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.4",
                            facecolor=fc, edgecolor=ec, linewidth=0.8, linestyle="--", zorder=1)
        ax.add_patch(bg)
        ax.text(x + w / 2, y + h - 1.1, lbl, ha="center", va="top",
                fontsize=7.1, fontweight="bold", color=C_GRAY, zorder=2)

    box(2.2, 23.0, 17.7, 10.0, "32 PDFs (641 pp.)",
        "Hybrid Index (v1 / Clean v2)\nWord-bbox tables + Markdown\npymupdf4llm reader, NFKC",
        fc="#FFFFFF", ec=C_NAVY)
    box(2.2, 11.8, 17.7, 9.2, "Visual & Page Catalog",
        "Figure/table page detector\nOn-demand 180 DPI render",
        fc="#FFFFFF", ec=C_TEAL, title_c=C_TEAL)
    box(2.2, 2.3, 17.7, 7.6, "32 Document Cards",
        "Structured entities, models,\nmetrics & year aliases",
        fc="#FFFFFF", ec=C_ORANGE, title_c=C_ORANGE)

    box(25.0, 20.0, 30.6, 13.0, "Path A: Pi Agent v2 (Thinking On)",
        "Hybrid BM25+BGE search, full-page Markdown reader\nSafe Python Decimal AST sandbox (cited operands)\nReasoning chain + evidence verifier (43.2k tok/q)",
        fc="#FFFFFF", ec=C_BLUE, title_c=C_BLUE)
    box(25.0, 3.8, 30.6, 13.0, "Path B: Multimodal VLM (mode2-only)",
        "Multi-query BM25 text pass + quote verification\nVisual escalation on abstentions (<=3 PDF pages)\nVisual answer + visual verify (12.8k tok/q)",
        fc="#FFFFFF", ec=C_TEAL, title_c=C_TEAL)

    box(60.8, 21.8, 21.0, 11.2, "Consensus Gate",
        "Cluster within +/-0.1% tol.\nCheck agreement &\ncitation validity",
        fc="#FFFFFF", ec=C_ORANGE, title_c=C_ORANGE)
    box(60.8, 3.8, 21.0, 12.2, "Tier-2: Doc Routing",
        "Card router -> Top-3 PDFs\nFull-doc Markdown reader\n+ 3-way arbitration",
        fc="#FFFFFF", ec=C_ORANGE, title_c=C_ORANGE)

    box(86.2, 11.0, 11.8, 17.6, "Verified CSV",
        "Unit scaling\n[low,high] norm\nRef ID -> URL\n5x is_blank\ncheck\n(79.7k tok/q)",
        fc="#FFFFFF", ec=C_TEAL, title_c=C_TEAL)

    arrow(19.9, 28.0, 25.0, 27.5, color=C_BLUE)
    arrow(19.9, 16.4, 25.0, 11.3, color=C_TEAL)
    arrow(19.9, 6.0, 60.8, 5.6, color=C_ORANGE, rad=0.06, ls="--")

    arrow(55.6, 27.5, 60.8, 27.5, color=C_BLUE)
    arrow(55.6, 12.3, 60.8, 24.5, color=C_TEAL)

    arrow(81.8, 27.5, 86.2, 23.0, label="76.6%\nagree", color=C_TEAL, lx=-0.2, ly=3.4)
    arrow(71.3, 21.8, 71.3, 16.0, label="23.4% disagree/blank", color=C_ORANGE, lx=0.0, ly=0.0)
    arrow(81.8, 9.8, 86.2, 15.2, color=C_ORANGE)

    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "fig1_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig2_pareto():
    """Zoomed Accuracy-Token Pareto Frontier on the 282-question Kaggle Private LB (0.71-0.91)."""
    fig, ax = plt.subplots(figsize=(3.45, 2.70))

    # (tokens, private, label, group, dx, dy, ha)
    points = [
        (10786, 0.735, "Pi v1", "baseline", 7, -4, "left"),
        (11001, 0.750, "v1 Chunk RAG", "baseline", 7, 2, "left"),
        (12848, 0.807, "Multimodal", "single", 9, -20, "left"),
        (15292, 0.815, "v2+MM Casc.", "single", 7, -11, "left"),
        (8708,  0.825, "Multi-hop RAG\n(8.7k)", "pareto", -2, 42, "left"),
        (22929, 0.818, "Pi v2 (off)", "single", 8, 3, "left"),
        (43182, 0.854, "Pi v2 (Think)", "pareto", 9, 2, "left"),
        (44485, 0.865, "Pure-RAG 3-way\n(44.5k, no Think)", "pareto", -10, 32, "right"),
        (79711, 0.872, "Main Pure-RAG\nCascade (79.7k)", "main", 8, 19, "center"),
        (86253, 0.841, "Doc Routing", "single", -6, -10, "right"),
        (125557, 0.845, "v1->FC Casc.", "fc", 6, -10, "left"),
        (689804, 0.815, "FC (raw)", "fc", -8, -4, "right"),
        (689804, 0.837, "FC + Recov.", "fc", -8, 6, "right"),
        (840795, 0.891, "5-Par.+FC\nCeiling (841k)", "ceiling", -10, 2, "right"),
    ]

    # Pareto frontier curve across RAG + ceiling operating points
    pareto_pts = [
        (8708, 0.825),
        (43182, 0.854),
        (44485, 0.865),
        (79711, 0.872),
        (840795, 0.891),
    ]
    px, py = zip(*pareto_pts)
    ax.plot(px, py, color=C_BLUE, linewidth=1.35, linestyle="-", alpha=0.85, zorder=2)

    # KohakuRAG 1st-place reference line
    ax.axhline(0.861, color=C_ORANGE, linestyle="--", linewidth=1.0, zorder=1)
    ax.text(1.12e6, 0.8625, "KohakuRAG 1st (0.861)", fontsize=6.3,
            color=C_ORANGE, fontweight="bold", va="bottom", ha="right")

    styles = {
        "baseline": dict(marker="o", color=C_GRAY, s=26, zorder=3),
        "single":   dict(marker="s", color="#4C9FD8", s=26, zorder=3),
        "fc":       dict(marker="^", color="#888888", s=30, zorder=3),
        "pareto":   dict(marker="D", color=C_BLUE, s=34, zorder=4),
        "main":     dict(marker="*", color=C_TEAL, s=110, edgecolor="#0B4F3A", linewidth=0.6, zorder=6),
        "ceiling":  dict(marker="P", color=C_PURPLE, s=48, zorder=5),
    }

    for x, y, lbl, grp, dx, dy, ha in points:
        ax.scatter([x], [y], **styles[grp])
        weight = "bold" if grp in ("main", "pareto", "ceiling") else "normal"
        col = C_TEAL if grp == "main" else (C_PURPLE if grp == "ceiling" else "#222222")
        ax.annotate(
            lbl, (x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=6.2, fontweight=weight, color=col, ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.45, shrinkA=0, shrinkB=2.5)
            if abs(dx) > 6 or abs(dy) > 8 else None,
        )

    ax.set_xscale("log")
    ax.set_xlim(4.8e3, 1.35e6)
    ax.set_ylim(0.715, 0.922)
    ax.set_xlabel("Mean Total Tokens / Question (log scale)")
    ax.set_ylabel("Kaggle Private Score (282 test Qs)")
    ax.grid(True, which="major", linestyle=":", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Compact legend in bottom-right empty space
    handles = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_TEAL, markeredgecolor="#0B4F3A", markersize=8, label="Main Pure-RAG Cascade"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=C_BLUE, markersize=5, label="Pure-RAG Pareto"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#888888", markersize=5.5, label="Full-Context Baselines"),
        plt.Line2D([0], [0], marker="P", color="w", markerfacecolor=C_PURPLE, markersize=6, label="5-Par. + FC Ceiling"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#DDDDDD", handletextpad=0.3, borderpad=0.35, fontsize=6.2)

    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / "fig2_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig3_ablations():
    """Three controlled test-set ablation panels on Kaggle Public & Private LB."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.08), sharey=True)

    panels = [
        (
            "(a) Index Cleaning & Multimodal",
            ["v1\nText", "v2 Clean\nText", "v1\nVLM", "v2+VLM\nCasc."],
            [0.814, 0.848, 0.854, 0.858],
            [0.750, 0.768, 0.807, 0.815],
        ),
        (
            "(b) Pi Agent & Thinking",
            ["Pi v1\nBaseline", "Pi v2\nThink Off", "Pi v2\nThink On"],
            [0.785, 0.873, 0.870],
            [0.735, 0.818, 0.854],
        ),
        (
            "(c) Recovery & Ensemble Design",
            ["FC\nRaw", "FC+\nRecov.", "7-Run\nCorr.", "5-Par\nQuor.", "5-Par\nBF", "5-Par\n+MM"],
            [0.858, 0.883, 0.896, 0.894, 0.900, 0.896],
            [0.815, 0.837, 0.865, 0.878, 0.888, 0.891],
        ),
    ]

    for ax, (title, labels, pub, priv) in zip(axes, panels):
        x = np.arange(len(labels))
        w = 0.36
        ax.bar(x - w / 2, pub, w, label="Public LB", color="#8CBCE6", edgecolor=C_NAVY, linewidth=0.5)
        b2 = ax.bar(x + w / 2, priv, w, label="Private LB", color=C_NAVY, edgecolor=C_NAVY, linewidth=0.5)
        ax.axhline(0.861, color=C_ORANGE, linestyle="--", linewidth=0.9, zorder=1, label="KohakuRAG (0.861)")
        ax.set_title(title, fontsize=7.8, fontweight="bold", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylim(0.70, 0.955)
        ax.grid(axis="y", linestyle=":", alpha=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for rect in b2:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=C_NAVY, rotation=90,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75))

    axes[0].set_ylabel("Kaggle Weighted Score (282 Qs)")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#DDDDDD",
                   fontsize=6.1, handlelength=1.4, borderpad=0.3)

    fig.tight_layout(pad=0.2, w_pad=0.5)
    fig.savefig(OUT / "fig3_ablations.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig4_cost_diversity():
    """Token costs and pairwise answer agreement on the 282-question test set."""
    fig = plt.figure(figsize=(7.0, 1.60))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0],
                           left=0.18, right=0.99, bottom=0.24, top=0.87, wspace=0.29)
    cost_grid = grid[0].subgridspec(1, 2, width_ratios=[1.8, 1], wspace=0.06)
    low = fig.add_subplot(cost_grid[0])
    high = fig.add_subplot(cost_grid[1], sharey=low)
    low.set_zorder(high.get_zorder() + 1)  # Keep the 151k label above the adjoining axis.
    heat = fig.add_subplot(grid[1])

    components = [
        ("MHop", C_GRAY), ("VLM", C_TEAL), ("Pi v2", C_BLUE),
        ("DocR", C_PURPLE), ("Full-Context", C_ORANGE),
    ]
    systems = [
        ("Multi-hop RAG", [8.7, 0, 0, 0, 0], "8.7k (0.825)"),
        ("Pure-RAG 3-way", [8.7, 12.8, 22.9, 0, 0], "44.5k (0.865)"),
        ("Main Pure-RAG Casc.", [0, 12.8, 43.2, 23.7, 0], "79.7k (0.872)"),
        ("4-Par. No-FC", [8.7, 12.8, 43.2, 86.3, 0], "151k (0.872)"),
        ("Full-Context (Recov.)", [0, 0, 0, 0, 689.8], "690k (0.837)"),
        ("5-Par. + FC Ceiling", [8.7, 12.8, 43.2, 86.3, 689.8], "841k (0.891)"),
    ]
    for row, (_, widths, annotation) in enumerate(systems):
        # Draw in cumulative coordinates and let each axis clip the same stack.
        start = 0
        order = [2, 1, 3] if row == 2 else range(len(components))
        for col in order:
            width = widths[col]
            if width:
                for ax in (low, high):
                    ax.barh(row, width, left=start, height=0.62, color=components[col][1])
                start += width
        ax = high if start >= 640 else low
        ax.annotate(annotation, (start, row), xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=6.5, color=C_NAVY)

    low.set_xlim(0, 175)
    high.set_xlim(640, 940)
    low.set_ylim(len(systems) - 0.5, -0.5)
    low.set_yticks(range(len(systems)), [name for name, _, _ in systems], fontsize=7.6)
    main_label = low.get_yticklabels()[2]
    main_label.set_color(C_TEAL)
    main_label.set_fontweight("bold")
    high.tick_params(axis="y", labelleft=False)
    low.set_xticks([0, 50, 100, 150])
    high.set_xticks([650, 800, 940])

    for ax, edge in ((low, 1), (high, 0)):
        ax.set_axisbelow(True)
        ax.grid(axis="x", linestyle=":", alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", labelsize=6.5, length=2, pad=3)
        ax.plot(edge, 0, transform=ax.transAxes, marker=[(-1, -1), (1, 1)],
                markersize=4.5, markeredgewidth=0.7, color=C_NAVY,
                linestyle="none", clip_on=False)
    high.spines["left"].set_visible(False)

    cost_box = grid[0].get_position(fig)
    cost_center = (cost_box.x0 + cost_box.x1) / 2
    fig.text(cost_center, grid.top + 0.045, "(a) Token Cost Breakdown (k tok/q) & Private LB",
             ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    fig.legend(handles=[Patch(facecolor=color, label=name) for name, color in components],
               loc="lower center", bbox_to_anchor=(cost_center, 0.035), ncol=5,
               frameon=False, fontsize=6.5, handlelength=1.0, handletextpad=0.35,
               columnspacing=0.85, borderaxespad=0)

    labels = ["Pi-On", "Pi-Off", "DocR", "FC-Rec", "MHop", "VLM", "v2+VLM"]
    # Test-set percentages from wattbot.ensemble.values_equivalent.
    agreement = np.array([
        [100.0, 90.8, 89.7, 85.1, 89.4, 82.6, 85.1],
        [90.8, 100.0, 87.9, 81.2, 87.6, 80.5, 83.3],
        [89.7, 87.9, 100.0, 88.7, 87.9, 83.3, 85.8],
        [85.1, 81.2, 88.7, 100.0, 83.7, 81.6, 83.3],
        [89.4, 87.6, 87.9, 83.7, 100.0, 84.4, 85.8],
        [82.6, 80.5, 83.3, 81.6, 84.4, 100.0, 90.8],
        [85.1, 83.3, 85.8, 83.3, 85.8, 90.8, 100.0],
    ])
    count = len(labels)
    diagonal = np.eye(count, dtype=bool)
    edges = np.arange(count + 1) - 0.5
    mesh = heat.pcolormesh(
        edges, edges, np.ma.array(agreement, mask=diagonal),
        cmap=plt.get_cmap("Blues").with_extremes(bad="#F0F2F5"),
        vmin=79, vmax=92, edgecolors="white", linewidth=0.45,
    )
    for (row, col), value in np.ndenumerate(agreement):
        red, green, blue, _ = mesh.cmap(mesh.norm(value))
        brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        color = C_NAVY if diagonal[row, col] or brightness > 0.55 else "white"
        heat.text(col, row, "—" if diagonal[row, col] else f"{value:.1f}",
                  ha="center", va="center", fontsize=6.5, color=color)
    for first in (0, 5):
        heat.add_patch(Rectangle((first - 0.45, first - 0.45), 1.9, 1.9, fill=False,
                                 edgecolor=C_ORANGE, linewidth=0.9, linestyle="--"))

    heat.invert_yaxis()
    heat.spines[:].set_visible(False)
    heat.set_xticks(range(count), labels, rotation=35, ha="right", fontsize=6.5)
    heat.set_yticks(range(count), labels, fontsize=6.5)
    heat.tick_params(length=0, pad=3)
    heat.set_title("(b) Pairwise Value Agreement (%, 282 Test Qs)",
                   fontsize=7.8, fontweight="bold", pad=9)
    heat.text(0.5, -0.28, "Dashed: correlated pairs (90.8%)", transform=heat.transAxes,
              ha="center", va="top", fontsize=6.5, color=C_ORANGE)

    fig.savefig(OUT / "fig4_cost_diversity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1_architecture()
    make_fig2_pareto()
    make_fig3_ablations()
    make_fig4_cost_diversity()
    print("Regenerated all 4 PDF figures in", OUT)
