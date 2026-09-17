"""
Helper functions for 04_disease_modules_optional.ipynb.

`degree_bins` is a plumbing detail of the permutation test (how we draw degree-matched random
node sets) rather than the statistical method itself; the plotting functions are just matplotlib
bookkeeping; `fetch_disease_genes` and `symbols_to_ncbi_in_network` are API/ID-mapping mechanics
this notebook needs but isn't teaching (Open Targets' GraphQL API, and gene-symbol lookup — the
latter is the actual lesson of Notebook 1, not this one). Both read from a precomputed `lookups/`
file by default (the same two reference diseases every workshop run compares against, and the
same PPI-network gene ID<->symbol table Notebook 3 uses) and only fall back to a live call for
whatever that file doesn't cover — e.g. if you've pointed DISEASE_SIMILAR/DISEASE_DIFFERENT at
your own diseases of interest. None of it is the lesson here, so it lives in this file instead of
the notebook. Nothing stops you from opening it and reading it.
"""
import json
import os

import numpy as np
import requests
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


def fetch_disease_genes(disease_name, lookups_dir=None, top_n=300, min_score=0.0):
    """A disease's Open Targets associated genes -- from the precomputed cache if `lookups_dir`
    is given and covers this exact disease_name (the two reference diseases this workshop
    compares against by default), otherwise a live Open Targets search + association lookup.
    Returns (efo_id, gene_symbols) — efo_id is None if no match was found."""
    if lookups_dir is not None:
        cache_path = os.path.join(lookups_dir, "opentargets_diseases.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f).get(disease_name)
            if cached:
                ranked = sorted(cached["gene_scores"], key=cached["gene_scores"].get, reverse=True)[:top_n]
                genes = {g for g in ranked if cached["gene_scores"][g] >= min_score}
                print(f"'{disease_name}' -> {cached['efo_id']} (cached, {len(genes)} associated genes)")
                return cached["efo_id"], genes

    print(f"'{disease_name}' not in the precomputed lookup -- querying Open Targets live")
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


def symbols_to_ncbi_in_network(symbols, G, lookups_dir=None):
    """Map gene symbols to NCBI Gene IDs and restrict to whichever of those IDs are actually
    nodes in the given network. Checks two precomputed tables first if `lookups_dir` is given:
    the PPI gene ID<->symbol table (same one Notebook 3 uses, for symbols that are already a
    network node's canonical symbol) and a small alias table covering every symbol in this
    workshop's two default reference diseases that *isn't* already covered by the first table
    (checked once via mygene.info, including the ones that don't resolve to a network node, so
    we already know not to ask again). Only a symbol in neither table -- i.e. you're comparing
    against your own disease of interest -- triggers a live mygene.info call."""
    network_ids = set(G.nodes())
    symbols = set(symbols)
    ids = set()

    symbol_to_id = {}
    checked_aliases = {}
    if lookups_dir is not None:
        import pandas as pd
        path = os.path.join(lookups_dir, "ppi_gene_id_symbol_lookup.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, dtype=str)
            symbol_to_id = dict(zip(df["symbol"], df["ncbi_gene_id"]))
        alias_path = os.path.join(lookups_dir, "disease_gene_symbol_matches.csv")
        if os.path.exists(alias_path):
            df = pd.read_csv(alias_path, dtype=str).fillna("")
            checked_aliases = dict(zip(df["query"], df["ncbi_gene_id"]))

    never_checked = []
    for s in symbols:
        gid = symbol_to_id.get(s)
        if gid:
            ids.add(gid)
        elif s in checked_aliases:
            if checked_aliases[s]:
                ids.add(checked_aliases[s])
            # else: already confirmed via mygene.info that this isn't a PPI network node
        else:
            never_checked.append(s)

    if never_checked:
        import mygene
        print(f"{len(never_checked)} symbol(s) not in the precomputed lookup -- querying mygene.info live")
        mg = mygene.MyGeneInfo()
        result = mg.querymany(never_checked, scopes="symbol,alias", fields="entrezgene",
                               species="human", returnall=True)
        for hit in result["out"]:
            if "entrezgene" in hit:
                ids.add(str(hit["entrezgene"]))

    return ids & network_ids


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
