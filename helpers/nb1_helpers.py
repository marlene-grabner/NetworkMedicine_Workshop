"""
Helper functions for 01_matching_and_loading.ipynb.

The mechanics behind two lookups -- not the matching logic itself (that's the actual lesson, and
stays in the notebook), just fetching/reshaping the reference data it matches against. Both read
from a small bundled `lookups/` file by default (precomputed once for the workshop's synthetic
data, so nobody's laptop hits an external API mid-workshop) and only fall back to a live request
if you've pointed this at your own data and the bundled file doesn't cover it. Nothing stops you
from opening this file and reading it if you're curious how it works.
"""
import os

import requests


def fetch_kegg_compound_names(lookups_dir):
    """KEGG's full compound list (~19k entries) as a name/synonym lookup.

    Reads `<lookups_dir>/kegg_compound_list.tsv` if it's there (the default -- precomputed once
    from KEGG's bulk `list/compound` endpoint, since this list doesn't depend on any particular
    DE data). Only fetched live, and cached to that path for next time, if the file is missing.

    Returns (kegg_id_to_names, name_to_kegg): the first maps a KEGG Compound ID to every name
    it's known by, the second flattens that into lowercase-name -> ID for exact-match lookups
    (first name wins if two compounds happen to share a synonym).
    """
    cache_path = os.path.join(lookups_dir, "kegg_compound_list.tsv")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            text = f.read()
    else:
        print("[live fetch] kegg_compound_list.tsv not found locally -- "
              "downloading KEGG's compound list directly (one-time, ~1MB)")
        resp = requests.get("https://rest.kegg.jp/list/compound", timeout=60)
        resp.raise_for_status()
        text = resp.text
        os.makedirs(lookups_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            f.write(text)

    kegg_id_to_names = {}
    for line in text.strip().split("\n"):
        cid, names = line.split("\t")
        cid = cid.replace("cpd:", "").strip()
        kegg_id_to_names[cid] = [n.strip() for n in names.split(";")]

    name_to_kegg = {}
    for cid, names in kegg_id_to_names.items():
        for n in names:
            name_to_kegg.setdefault(n.lower(), cid)

    return kegg_id_to_names, name_to_kegg


def load_symbol_lookup(lookups_dir):
    """Precomputed symbol/alias -> NCBI Gene ID matches for this workshop's DEGs.csv and
    DE_proteins.csv symbols (mygene.info, checked once for every symbol in both files -- so a
    symbol simply not being in this table means it wasn't part of the original workshop data,
    not that it failed to match). Returns {symbol: (ncbi_gene_id, matched_symbol) | None} --
    None means mygene.info was already asked about this exact symbol and had no hit, so the
    notebook can go straight to fuzzy matching instead of asking again. A symbol missing
    entirely from the returned dict is new (e.g. you edited DEGs.csv/DE_proteins.csv yourself)
    and falls through to the notebook's live-fallback path. Returns an empty dict if the file
    isn't there at all (e.g. a from-scratch custom setup)."""
    path = os.path.join(lookups_dir, "deg_protein_symbol_matches.csv")
    if not os.path.exists(path):
        return {}
    import pandas as pd
    df = pd.read_csv(path, dtype=str).fillna("")
    lookup = {}
    for _, row in df.iterrows():
        lookup[row["query"]] = (row["ncbi_gene_id"], row["matched_symbol"]) if row["ncbi_gene_id"] else None
    return lookup
