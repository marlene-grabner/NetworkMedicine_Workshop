"""
Helper functions for 03_bridging.ipynb.

Four kinds of thing live here, none of them the actual network-medicine lesson: loading the
precomputed lookups this notebook reads by default (a gene ID<->symbol table, cached Open Targets
results) — plus the *live* Open Targets calls those lookups stand in for, kept available for
anyone who points this at their own disease of interest; the graph-algorithm mechanics behind the
prize-collecting bridge module (the degree-penalized cost graph, turning terminals into a
minimum-cost Steiner tree, and the prize-collecting pruning that decides which of them were worth
it — standard graph-theory machinery, not something specific to this analysis); the degree-matched
null-model permutation loop that validates the resulting module; and the drawing mechanics behind
the final figure (tiers, bands, labels, legend). The notebook itself keeps the actual *decisions*
— prizes, costs, how strict the pruning is, how many hops the figure shows — as tunable parameters
up top. Nothing stops you from opening this file and reading it if you're curious how any of it
works.
"""

import json
import os
import time

import numpy as np
import requests
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"


def load_gene_symbol_lookup(lookups_dir):
    """Precomputed NCBI Gene ID -> symbol table, covering every node in the PPI network (built
    once via mygene.info). Used to label connector genes without a live lookup per gene. Returns
    an empty dict if the file isn't there, in which case labels just fall back to the raw ID."""
    path = os.path.join(lookups_dir, "ppi_gene_id_symbol_lookup.csv")
    if not os.path.exists(path):
        return {}
    import pandas as pd
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["ncbi_gene_id"], df["symbol"]))


def load_cached_enrichment(lookups_dir, cache_key):
    """A precomputed g:Profiler enrichment table for one of this notebook's two default queries
    (see lookups/<cache_key>.csv, built once from the workshop's own synthetic data). Returns
    None if that file isn't there, so the caller can fall back to a live g:Profiler call."""
    path = os.path.join(lookups_dir, f"{cache_key}.csv")
    if not os.path.exists(path):
        return None
    import pandas as pd
    return pd.read_csv(path)


def load_cached_disease(lookups_dir, disease_name):
    """A precomputed Open Targets result (search hits, EFO ID, associated gene symbols) for one
    of the three disease queries this workshop runs by default -- see
    lookups/opentargets_diseases.json. `disease_name` is matched exactly against the query string
    used when the lookup was built (e.g. "chronic kidney disease"). Returns None if the file or
    that exact query isn't cached, so the caller can fall back to a live search_disease() /
    fetch_disease_targets() call -- e.g. if you've changed DISEASE_QUERY to your own phenotype."""
    path = os.path.join(lookups_dir, "opentargets_diseases.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        all_diseases = json.load(f)
    return all_diseases.get(disease_name)


def search_disease(query, size=5):
    """Search Open Targets for a disease by free-text name, returning the raw list of hits
    (each with an `id` you can pass to fetch_disease_targets)."""
    q = """
    query searchDisease($q: String!, $size: Int!) {
      search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: $size}) {
        hits { id name entity }
      }
    }
    """
    try:
        r = requests.post(
            OT_API,
            json={"query": q, "variables": {"q": query, "size": size}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"]["search"]["hits"]
    except Exception as e:
        print(f"[warn] Open Targets search failed: {e}")
        return []


def fetch_disease_targets(efo_id, size=500):
    """Every gene symbol Open Targets associates with a disease, given its EFO/MONDO ID."""
    q = """
    query diseaseTargets($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        associatedTargets(page: {index: 0, size: $size}) {
          rows { target { id approvedSymbol } score }
        }
      }
    }
    """
    try:
        r = requests.post(
            OT_API,
            json={"query": q, "variables": {"efoId": efo_id, "size": size}},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()["data"]["disease"]["associatedTargets"]["rows"]
        return {row["target"]["approvedSymbol"] for row in rows}
    except Exception as e:
        print(f"[warn] Open Targets association query failed: {e}")
        return set()


def build_cost_graph(G):
    """Turn G into a sparse adjacency matrix weighted by a degree-penalized edge cost —
    sqrt(deg(u) * deg(v)), normalized so the network's median edge costs about 1 — so a shortest
    "cheapest path" through it prefers specific, low-degree interactions over generic hub
    proteins. Returns (nodes, node_idx, A, degrees) for use with terminal_steiner_tree."""
    degrees = dict(G.degree())
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    raw = np.array([np.sqrt(degrees[u] * degrees[v]) for u, v in G.edges()])
    median_cost = np.median(raw) if len(raw) else 1.0

    rows, cols, data = [], [], []
    for u, v in G.edges():
        c = np.sqrt(degrees[u] * degrees[v]) / median_cost
        ui, vi = node_idx[u], node_idx[v]
        rows += [ui, vi]
        cols += [vi, ui]
        data += [c, c]
    A = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    return nodes, node_idx, A, degrees


def terminal_steiner_tree(nodes, node_idx, A, terminals):
    """The minimum-cost tree connecting every node in `terminals` through the graph behind A:
    compute cost-shortest paths between every pair of terminals, take the minimum spanning tree
    of that terminal-to-terminal distance matrix, then stitch each MST edge back into the real
    shortest path it stands for. This is the standard "MST of the metric closure" 2-approximation
    to a Steiner tree — it forces every terminal in, with no judgment yet about whether a given
    terminal was worth including (that's the prize-collecting step, back in the notebook).
    Returns an nx.Graph with each edge's cost stored as edge attribute "cost"."""
    term_idx = [node_idx[t] for t in terminals]
    dist_matrix, predecessors = dijkstra(
        csgraph=A, directed=False, indices=term_idx, return_predecessors=True
    )
    D = dist_matrix[:, term_idx]

    term_graph = nx.Graph()
    term_graph.add_nodes_from(terminals)
    for i in range(len(terminals)):
        for j in range(i + 1, len(terminals)):
            if np.isfinite(D[i, j]):
                term_graph.add_edge(terminals[i], terminals[j], weight=D[i, j])
    mst = nx.minimum_spanning_tree(term_graph, weight="weight")

    tree = nx.Graph()
    for u, v in mst.edges():
        src_row = term_idx.index(node_idx[u])
        path = [node_idx[v]]
        cur = node_idx[v]
        while predecessors[src_row, cur] != -9999:
            cur = predecessors[src_row, cur]
            path.append(cur)
        path = path[::-1]
        for a, b in zip(path[:-1], path[1:]):
            na, nb = nodes[a], nodes[b]
            tree.add_edge(na, nb, cost=A[a, b])
    return tree


def select_metabolite_terminals(cost_nodes, cost_idx, A_cost, protein_terminals, metab_to_genes):
    """One terminal gene per metabolite: whichever of its candidate bridging genes is cheapest to
    reach from the protein module — lowest degree-penalized cost, not just fewest hops.
    Returns (metab_terminal_gene, metab_terminals): a {kegg_id: gene} map, and the set of its
    (deduplicated) values."""
    dist_from_module = dijkstra(
        csgraph=A_cost, directed=False,
        indices=[cost_idx[n] for n in protein_terminals], min_only=True,
    )
    metab_terminal_gene = {
        cid: min(genes_here, key=lambda g: dist_from_module[cost_idx[g]])
        for cid, genes_here in metab_to_genes.items()
    }
    return metab_terminal_gene, set(metab_terminal_gene.values())


def _prune_module(tree, terminal_set, prize, strictness):
    """The prize-collecting step: repeatedly trim dangling branch tips that aren't worth their
    cost. A non-terminal (Steiner) leaf is always dangling — it exists only to connect something
    else, so once it's down to degree 1 it's connecting nothing and gets cut. A terminal leaf is
    only cut if the single edge holding it to the tree costs more than its own prize (times
    `strictness`) — i.e. we'd rather drop it than pay that much just to include it."""
    tree = tree.copy()
    changed = True
    while changed:
        changed = False
        for leaf in [n for n in tree.nodes() if tree.degree(n) == 1]:
            if tree.number_of_nodes() <= 1:
                break
            nbr = next(iter(tree.neighbors(leaf)))
            edge_cost = tree.edges[leaf, nbr]["cost"]
            if leaf not in terminal_set or edge_cost > prize.get(leaf, 0) * strictness:
                tree.remove_node(leaf)
                changed = True
    return tree


def build_bridge_module(cost_nodes, cost_idx, A_cost, protein_terms, metab_terms, prize, strictness=1.0):
    """Force every terminal in via a minimum-cost Steiner tree, then prune away the branches that
    weren't worth their cost — the full prize-collecting Steiner tree build, in one call. Returns
    the final module as an nx.Graph, each edge carrying its cost."""
    all_terms = protein_terms | metab_terms
    raw_tree = terminal_steiner_tree(cost_nodes, cost_idx, A_cost, list(all_terms))
    return _prune_module(raw_tree, all_terms, prize, strictness)


def degree_matched_sample(G, seed_nodes, rng, n_bins=10):
    """Draw a random node set the same size as seed_nodes, matched bin-for-bin on degree — the
    same idea as Notebook 4's null model, used here to ask whether a same-size, similarly-
    connected random set of genes would build as cheap a bridge module as the real one does."""
    degrees = dict(G.degree())
    deg_values = np.array(list(degrees.values()))
    edges = np.unique(np.quantile(deg_values, np.linspace(0, 1, n_bins + 1)))
    nodes_in_bin = {i: [] for i in range(len(edges) - 1)}
    for node, d in degrees.items():
        b = min(np.searchsorted(edges, d, side="right") - 1, len(edges) - 2)
        nodes_in_bin[b].append(node)

    seed_set = set(seed_nodes)
    sample = set()
    for n in seed_nodes:
        d = degrees[n]
        b = min(np.searchsorted(edges, d, side="right") - 1, len(edges) - 2)
        pool = [x for x in nodes_in_bin[b] if x not in seed_set and x not in sample]
        sample.add(rng.choice(pool) if pool else rng.choice(list(G.nodes())))
    return sample


def validate_bridge_module(G, cost_nodes, cost_idx, A_cost, protein_terminals, metab_terminals,
                            observed_cost, n_permutations=100, strictness=1.0, seed=0):
    """Null-model validation (Menche et al. 2015 style): rebuild the same prize-collecting bridge
    module `n_permutations` times from degree-matched random terminal sets the same sizes as the
    real ones, and compare the real module's total cost to that null distribution. A genuine,
    specific mechanism should be cheaper to connect (fewer hops, lower-degree interactions) than
    an arbitrary same-size gene set — so a negative z-score is the result you're hoping for.
    Returns (null_costs, z_score, p_value)."""
    rng = np.random.default_rng(seed)
    null_costs = []
    t0 = time.time()
    for _ in range(n_permutations):
        rand_proteins = degree_matched_sample(G, protein_terminals, rng)
        rand_metabs = degree_matched_sample(G, metab_terminals, rng)
        rand_terms = rand_proteins | rand_metabs
        rand_tree = build_bridge_module(cost_nodes, cost_idx, A_cost, rand_proteins, rand_metabs,
                                         prize={n: 1.0 for n in rand_terms}, strictness=strictness)
        null_costs.append(sum(d["cost"] for _, _, d in rand_tree.edges(data=True)))
    print(f"{n_permutations} permutations in {time.time() - t0:.0f}s")

    null_costs = np.array(null_costs)
    z_score = (observed_cost - null_costs.mean()) / null_costs.std()
    p_value = (null_costs <= observed_cost).mean()   # one-sided: how often does chance alone build something this cheap?
    return null_costs, z_score, p_value


def layered_layout(G, tiers, spacing=1.3, tier_gap=5.0, n_iter=4):
    """Lay nodes out in vertical tiers left to right, using a barycenter heuristic — a few
    passes of nudging each node towards the average height of its neighbors in the *other*
    tiers, then re-spacing evenly in that order — to keep connected nodes roughly level and cut
    down on crossing edges, instead of just leaving every node at an arbitrary position within
    its tier."""
    order = [sorted(tier) for tier in tiers]

    def positions_from_order(order):
        pos = {}
        for tier_idx, tier_nodes in enumerate(order):
            n = len(tier_nodes)
            for i, node in enumerate(tier_nodes):
                pos[node] = (tier_idx * tier_gap, (i - (n - 1) / 2.0) * spacing)
        return pos

    pos = positions_from_order(order)
    for _ in range(n_iter):
        for tier_idx, tier_nodes in enumerate(order):
            if not tier_nodes:
                continue
            barycenter = {}
            for node in tier_nodes:
                ys = [
                    pos[nb][1]
                    for nb in G.neighbors(node)
                    if nb in pos and pos[nb][0] != tier_idx * tier_gap
                ]
                barycenter[node] = sum(ys) / len(ys) if ys else pos[node][1]
            order[tier_idx] = sorted(tier_nodes, key=lambda n: barycenter[n])
            pos = positions_from_order(order)
    return pos


def plot_layered_bridge(
    viz_graph,
    layout_tiers,
    style_groups,
    labels,
    tier_labels,
    legend_entries,
    title,
    save_path,
    tier_gap=6.0,
    spacing=1.3,
    label_fontsize=8,
):
    """Draw a multi-tier bridge figure — proteins on the left, metabolites on the right,
    connector genes tiered in between — and save it.

    layout_tiers: node-sets left to right, used to compute positions (e.g. [proteins,
        1-hop connectors, 2-hop connectors, metabolites]) — every node must appear in exactly
        one of these.
    style_groups: (nodes, color, marker_shape, marker_size) tuples for drawing — can subdivide
        a layout tier into differently-styled groups.
    labels: {node: display_name} for every node.
    tier_labels: (name, color) per layout tier — drawn as a bold header over a light rounded
        band behind that tier, so the figure carries its own group labels instead of leaning on
        the legend for that.
    legend_entries: (label, color, marker_shape, marker_size) tuples for the (now much smaller)
        legend — tier identity lives in the headers, so this only needs to cover what the
        headers don't, e.g. marker shape.
    """
    pos = layered_layout(viz_graph, layout_tiers, spacing=spacing, tier_gap=tier_gap)
    n_tiers = len(layout_tiers)

    fig_height = max(9, 0.35 * max((len(t) for t in layout_tiers), default=1))
    fig, ax = plt.subplots(figsize=(15, fig_height))

    # A light rounded band behind each tier, headed by a bold label — drawn first so the nodes,
    # edges and labels all sit on top of it.
    band_pad = max(spacing, 1.0)
    for tier_idx, (tier_nodes, (tname, tcolor)) in enumerate(
        zip(layout_tiers, tier_labels)
    ):
        if not tier_nodes:
            continue
        x_c = tier_idx * tier_gap
        ys = [pos[n][1] for n in tier_nodes]
        y0, y1 = min(ys) - band_pad, max(ys) + band_pad
        band = FancyBboxPatch(
            (x_c - tier_gap * 0.34, y0),
            tier_gap * 0.68,
            y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.6",
            linewidth=1.2,
            edgecolor=tcolor,
            facecolor=tcolor,
            alpha=0.12,
        )
        ax.add_patch(band)
        ax.text(
            x_c,
            y1 + 0.5,
            tname,
            fontsize=12,
            fontweight="bold",
            color=tcolor,
            ha="center",
            va="bottom",
        )

    nx.draw_networkx_edges(viz_graph, pos, ax=ax, alpha=0.25, width=0.6)
    for nodes, color, shape, size in style_groups:
        nx.draw_networkx_nodes(
            viz_graph,
            pos,
            ax=ax,
            nodelist=list(nodes),
            node_color=color,
            node_shape=shape,
            node_size=size,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.5,
        )

    # Labels: horizontal text, pushed left of nodes in the left-hand tiers and right of nodes in
    # the right-hand tiers — same "push outward, away from the rest of the figure" idea as
    # before, just turned 90 degrees along with everything else, so plain horizontal text works
    # instead of needing to rotate it.
    for tier_idx, tier_nodes in enumerate(layout_tiers):
        dx, ha = (-0.3, "right") if tier_idx < n_tiers / 2 else (0.3, "left")
        for n in tier_nodes:
            x, y = pos[n]
            ax.text(
                x + dx, y, labels.get(n, n), fontsize=label_fontsize, ha=ha, va="center"
            )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=shape,
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=size,
        )
        for label, color, shape, size in legend_entries
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="#888888",
        title="Node shape",
        title_fontsize=10,
    )

    # Text doesn't count towards matplotlib's auto-scaling, so left alone the header labels and
    # the outermost name columns would get clipped in the inline view (the saved PNG is fine
    # either way, via bbox_inches="tight") — give the axes explicit headroom.
    x_vals = [x for x, _ in pos.values()]
    y_vals = [y for _, y in pos.values()]
    ax.set_xlim(min(x_vals) - tier_gap * 0.55, max(x_vals) + tier_gap * 0.55)
    ax.set_ylim(min(y_vals) - band_pad - 1.5, max(y_vals) + band_pad + 1.5)

    ax.set_title(title, pad=14, fontsize=13)
    ax.axis("off")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_bridge_flowchart(bridge_tree, protein_terminals, bridge_module_nodes, metab_terminal_gene,
                           metabs_matched, id_to_symbol, sym_lookup,
                           max_hop_cap=3, explosion_threshold=70, save_path=None):
    """Everything Step 6 needs beyond its two tunable parameters: hop-tier the connector genes by
    distance from the protein module within the bridge tree, cap the figure's depth if that would
    make it unreadably large, attach each shown metabolite to its own bridging gene, then hand
    off to plot_layered_bridge() for the actual drawing.

    max_hop_cap: draw at most this many hop-tiers ("direct" = 1, "one in-between hop" = 2, ...).
    explosion_threshold: total connector genes allowed across all drawn hop-tiers before the
        figure automatically falls back to fewer hops.
    """
    hop_from_protein = nx.multi_source_dijkstra_path_length(
        bridge_tree, protein_terminals & bridge_module_nodes, weight=lambda u, v, d: 1
    )

    max_hop = max_hop_cap
    while max_hop > 1 and sum(1 for h in hop_from_protein.values() if 1 <= h <= max_hop) > explosion_threshold:
        max_hop -= 1
    if max_hop < max_hop_cap:
        n_at_full_depth = sum(1 for h in hop_from_protein.values() if 1 <= h <= max_hop_cap)
        print(f"Capping the figure at {max_hop} hop(s) from the protein module to keep it readable "
              f"({n_at_full_depth} connector genes would appear at up to {max_hop_cap} hops).")

    hop_tiers = [{n for n, h in hop_from_protein.items() if h == hop} for hop in range(1, max_hop + 1)]

    # Only show a metabolite if its own bridging gene survived pruning and lies within the hop cap.
    shown_metab_ids = {cid for cid, gene in metab_terminal_gene.items()
                        if gene in bridge_module_nodes and hop_from_protein.get(gene, 999) <= max_hop}
    n_excluded = len(metab_terminal_gene) - len(shown_metab_ids)
    if n_excluded:
        print(f"{n_excluded} metabolite(s) not pictured -- their bridge lies beyond {max_hop} hop(s) "
              f"from the protein module, or was pruned in Step 3.")

    bottom_nodes = protein_terminals & bridge_module_nodes
    top_nodes = shown_metab_ids
    layout_tiers = [bottom_nodes] + hop_tiers + [top_nodes]
    visible_nodes = bottom_nodes | (set().union(*hop_tiers) if hop_tiers else set())

    viz_graph = nx.Graph()
    for tier in layout_tiers:
        viz_graph.add_nodes_from(tier)
    viz_graph.add_edges_from((u, v) for u, v in bridge_tree.edges() if u in visible_nodes and v in visible_nodes)
    for cid in top_nodes:
        viz_graph.add_edge(cid, metab_terminal_gene[cid])

    kegg_to_name = dict(zip(metabs_matched["kegg_id"], metabs_matched["matched_name"]))
    gene_label_lookup = {**id_to_symbol, **sym_lookup}
    # any node still unresolved here just displays as its raw NCBI Gene ID -- id_to_symbol and
    # sym_lookup already draw on the full precomputed PPI gene-symbol table, so there's nothing
    # left worth a live lookup for.
    labels = {cid: kegg_to_name.get(cid, cid) for cid in top_nodes}
    labels.update({n: gene_label_lookup.get(n, n) for n in visible_nodes})

    middle_colors = ["#8172B2", "#B3A9D9", "#DCD6EF"][:max_hop]
    style_groups = [(bottom_nodes, "#4C72B0", "o", 200)]
    style_groups += [(tier, color, "o", 220) for tier, color in zip(hop_tiers, middle_colors)]
    style_groups.append((top_nodes, "#55A868", "s", 200))

    tier_labels = [("Proteins", "#4C72B0")]
    tier_labels += [(f"{h} hop{'s' if h > 1 else ''}", color)
                     for h, color in zip(range(1, max_hop + 1), middle_colors)]
    tier_labels.append(("Metabolites", "#55A868"))

    legend_entries = [
        ("Gene (protein or connector)", "#888888", "o", 9),
        ("Metabolite", "#888888", "s", 9),
    ]

    plot_layered_bridge(
        viz_graph, layout_tiers=layout_tiers, style_groups=style_groups, labels=labels,
        tier_labels=tier_labels, legend_entries=legend_entries,
        title="Cross-omics bridge module — C3 glomerulopathy", save_path=save_path,
    )
