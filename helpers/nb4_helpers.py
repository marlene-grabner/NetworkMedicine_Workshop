"""
Helper functions for 04_disease_modules_optional.ipynb.

`degree_bins` is a plumbing detail of the permutation test (how we draw degree-matched random
node sets) rather than the statistical method itself; the plotting functions are just matplotlib
bookkeeping; `fetch_disease_genes` and `symbols_to_ncbi_in_network` are API/ID-mapping mechanics
this notebook needs but isn't teaching (Open Targets' GraphQL API, and gene-symbol lookup — the
latter is the actual lesson of Notebook 1, not this one). None of it is the lesson here, so it
lives in this file instead of the notebook. Nothing stops you from opening it and reading it.
"""
import os

import numpy as np
import requests
import mygene
import matplotlib.pyplot as plt

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"

SEARCH_QUERY = """
query searchDisease($q: String!) {
  search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 5}) {
    hits { id name entity }
  }
}
"""

ASSOC_QUERY = """
query diseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    associatedTargets(page: {index: 0, size: $size}) {
      rows { target { id approvedSymbol } score }
    }
  }
}
"""


def fetch_disease_genes(disease_name, top_n=300, min_score=0.0):
    """Look a disease up on Open Targets by name, then pull every gene it's associated with.
    Returns (efo_id, gene_symbols) — efo_id is None if no match was found."""
    r = requests.post(OT_API, json={"query": SEARCH_QUERY, "variables": {"q": disease_name}}, timeout=30)
    r.raise_for_status()
    hits = r.json()["data"]["search"]["hits"]
    if not hits:
        print(f"[warn] no disease match for '{disease_name}'")
        return None, set()
    efo_id = hits[0]["id"]
    print(f"'{disease_name}' -> {efo_id} ({hits[0]['name']})")

    r = requests.post(OT_API, json={"query": ASSOC_QUERY,
                                      "variables": {"efoId": efo_id, "size": top_n}}, timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]["disease"]["associatedTargets"]["rows"]
    genes = {row["target"]["approvedSymbol"] for row in rows if row["score"] >= min_score}
    print(f"  -> {len(genes)} associated genes (score >= {min_score})")
    return efo_id, genes


def symbols_to_ncbi_in_network(symbols, G):
    """Batch-map gene symbols to NCBI Gene IDs (mygene.info, same idea as Notebook 1's matching
    step) and restrict to whichever of those IDs are actually nodes in the given network."""
    mg = mygene.MyGeneInfo()
    result = mg.querymany(list(symbols), scopes="symbol,alias", fields="entrezgene",
                           species="human", returnall=True)
    ids = set()
    for hit in result["out"]:
        if "entrezgene" in hit:
            ids.add(str(hit["entrezgene"]))
    return ids & set(G.nodes())


def degree_bins(G, n_bins=10):
    """Group network nodes into buckets of similar degree, so a random comparison set can be
    drawn with a similar degree distribution to the real seed nodes — otherwise a random set
    would look artificially poorly connected just because hub nodes are rare."""
    degrees = dict(G.degree())
    deg_values = np.array(list(degrees.values()))
    edges = np.unique(np.quantile(deg_values, np.linspace(0, 1, n_bins + 1)))
    bin_of_node, nodes_in_bin = {}, {i: [] for i in range(len(edges) - 1)}
    for node, d in degrees.items():
        b = min(np.searchsorted(edges, d, side="right") - 1, len(edges) - 2)
        bin_of_node[node] = b
        nodes_in_bin[b].append(node)
    return bin_of_node, nodes_in_bin


def plot_connectivity_test(connectivity_results, proc_dir):
    """One histogram per network layer: the null distribution of largest-connected-component
    sizes from randomly drawn degree-matched node sets, with the real observed value marked."""
    fig, axes = plt.subplots(1, len(connectivity_results), figsize=(5 * len(connectivity_results), 4))
    if len(connectivity_results) == 1:
        axes = [axes]

    for ax, (layer, res) in zip(axes, connectivity_results.items()):
        ax.hist(res["null_dist"], bins=8, color="lightgray", edgecolor="white")
        ax.axvline(res["observed_lcc"], color="crimson", linewidth=2, label="observed")
        ax.set_title(f"{layer}\nz={res['z_score']:.2f}, p={res['p_value']:.4f}")
        ax.set_xlabel("largest connected component size")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(proc_dir, "connectivity_test.png"), dpi=150)
    plt.show()


def plot_s_ab_comparison(labels, values, proc_dir):
    """Bar chart of network separation S_AB for each reference disease — negative (green) means
    close to your module in the network, positive (red) means topologically separated."""
    plt.figure(figsize=(5, 4))
    colors = ["#55A868" if v < 0 else "#C44E52" for v in values]
    plt.bar(labels, values, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Network separation S$_{AB}$")
    plt.title("Your module vs. reference diseases")
    plt.xticks(rotation=15)
    for i, v in enumerate(values):
        plt.text(i, v + (0.02 if v >= 0 else -0.06), f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(proc_dir, "s_ab_comparison.png"), dpi=150)
    plt.show()
