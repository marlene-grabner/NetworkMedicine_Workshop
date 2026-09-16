"""
Helper functions for 01_matching_and_loading.ipynb.

Just the raw KEGG API call + text parsing behind the metabolite name lookup — not the matching
logic itself (that's the actual lesson, and stays in the notebook), just the mechanics of
fetching and reshaping KEGG's compound list. Nothing stops you from opening this file and
reading it if you're curious how it works.
"""
import requests


def fetch_kegg_compound_names():
    """Download KEGG's full compound list once (~19k entries, one request) and build a
    name/synonym lookup.

    Returns (kegg_id_to_names, name_to_kegg): the first maps a KEGG Compound ID to every name
    it's known by, the second flattens that into lowercase-name -> ID for exact-match lookups
    (first name wins if two compounds happen to share a synonym).
    """
    resp = requests.get("https://rest.kegg.jp/list/compound", timeout=60)
    resp.raise_for_status()

    kegg_id_to_names = {}
    for line in resp.text.strip().split("\n"):
        cid, names = line.split("\t")
        cid = cid.replace("cpd:", "").strip()
        kegg_id_to_names[cid] = [n.strip() for n in names.split(";")]

    name_to_kegg = {}
    for cid, names in kegg_id_to_names.items():
        for n in names:
            name_to_kegg.setdefault(n.lower(), cid)

    return kegg_id_to_names, name_to_kegg
