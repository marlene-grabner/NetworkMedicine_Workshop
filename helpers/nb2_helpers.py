"""
Helper functions for 02_overlay_enrichment.ipynb.

This file holds the plotting/layout mechanics behind the "module overview" figure — none of it
is specific to network medicine, it's just the bookkeeping needed to draw a module as a tight,
localized cluster with its network context pushed out around it. It's pulled out here so the
notebook itself can stay focused on the actual analysis. Nothing stops you from opening this
file and reading it if you're curious how the figure is built — it's plain Python.
"""
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# dark = module (largest connected component), light = real network neighbors shown for context
LAYER_COLORS = {
    "transcriptome": {"dark": "#1B7837", "light": "#A6DBA0"},   # green
    "ppi":           {"dark": "#2166AC", "light": "#92C5DE"},   # blue
    "metabolite":    {"dark": "#B2182B", "light": "#F4A582"},   # dark red
}
LAYER_TITLES = {"transcriptome": "Transcripts", "ppi": "Proteins", "metabolite": "Metabolites"}


def localized_layout(G, core, periphery, core_radius=0.35, periph_radius=1.0, seed=0):
    """Lay the module out as a tight cluster at the center, then place every context node near
    whichever module node(s) it actually connects to — so radial distance reflects real
    attachment to the module, instead of a force layout blurring the two together."""
    rng = np.random.default_rng(seed)
    core_sub = G.subgraph(core)
    core_pos = nx.spring_layout(core_sub, seed=seed, k=0.6) if core_sub.number_of_edges() else \
        {n: (np.cos(a), np.sin(a)) for n, a in zip(core, np.linspace(0, 2 * np.pi, len(core), endpoint=False))}

    xs = np.array([p[0] for p in core_pos.values()])
    ys = np.array([p[1] for p in core_pos.values()])
    cx, cy = xs.mean(), ys.mean()
    max_r = max(np.hypot(xs - cx, ys - cy).max(), 1e-6)
    core_pos = {n: ((x - cx) / max_r * core_radius, (y - cy) / max_r * core_radius) for n, (x, y) in core_pos.items()}

    periph_pos = {}
    for n in periphery:
        anchors = [core_pos[nb] for nb in G.neighbors(n) if nb in core_pos]
        if anchors:
            ax_, ay_ = np.mean([a[0] for a in anchors]), np.mean([a[1] for a in anchors])
            angle = np.arctan2(ay_, ax_)
        else:
            angle = rng.uniform(0, 2 * np.pi)
        angle += rng.normal(0, 0.12)   # jitter so nodes sharing an anchor don't stack exactly
        r = periph_radius * (1 + rng.uniform(-0.05, 0.05))
        periph_pos[n] = (r * np.cos(angle), r * np.sin(angle))

    return {**core_pos, **periph_pos}


def _top_context_nodes(G, lcc, n_context):
    """Real neighbors of the module that aren't part of it themselves, favoring whichever ones
    plug into the module the most."""
    neighbor_counts = {}
    for n in lcc:
        for nb in G.neighbors(n):
            if nb not in lcc:
                neighbor_counts[nb] = neighbor_counts.get(nb, 0) + 1
    return sorted(neighbor_counts, key=neighbor_counts.get, reverse=True)[:n_context]


def plot_module_overview(graphs, module_seed_nodes, module_lcc_nodes, layers, proc_dir, n_context=30):
    """One row of panels, one per network layer: each layer's module (dark) plotted together
    with a bit of its surrounding network (light), saved to <proc_dir>/module_overview.png."""
    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 6))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        G = graphs[layer]
        lcc = module_lcc_nodes[layer]
        colors = LAYER_COLORS[layer]

        if not lcc:
            ax.set_title(f"{LAYER_TITLES[layer]}\n(no connected module)")
            ax.axis("off")
            continue

        context_nodes = _top_context_nodes(G, lcc, n_context)
        sub = G.subgraph(set(lcc) | set(context_nodes))
        pos = localized_layout(sub, lcc, context_nodes)

        node_colors = [colors["dark"] if n in lcc else colors["light"] for n in sub.nodes()]
        node_sizes = [110 if n in lcc else 35 for n in sub.nodes()]

        ax.add_patch(plt.Circle((0, 0), 0.35 * 1.2, fill=False, linestyle="--",
                                 edgecolor=colors["dark"], alpha=0.4, linewidth=1))
        nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.25, width=0.7)
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                                linewidths=0.4, edgecolors="white")
        ax.set_title(f"{LAYER_TITLES[layer]}\nmodule (dark, n={len(lcc)}) + network context (light, n={len(context_nodes)})",
                     fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(proc_dir, "module_overview.png"), dpi=150, bbox_inches="tight")
    plt.show()
