# Network Medicine Workshop — Kidney Disease

Integrating spatial transcriptomics, proteomics, and metabolomics via network medicine.

## For attendees — zero setup

You need **a free Google account and a browser**. Nothing to install, nothing to upload.

1. Click **Notebook 01** below — it opens live in Google Colab
2. Click **"Connect"** when it asks about Google Drive (one click — this is just where your results get saved between notebooks, not something you need to prepare)
3. Run the cells top to bottom (`Runtime → Run all` also works). Step 0 in Notebook 1 automatically downloads all the data and network files for you — nothing to fetch or upload yourself
4. Continue to Notebooks 02 and 03 in order. Notebook 04 is optional

If you want to keep an edited copy of a notebook, `File → Save a copy in Drive` at any point — this never affects the original.

| Notebook | Opens in Colab |
|---|---|
| 01 — Matching & Loading | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo>/blob/main/notebooks/01_matching_and_loading.ipynb) |
| 02 — Overlay & Enrichment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo>/blob/main/notebooks/02_overlay_enrichment.ipynb) |
| 03 — Bridging | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo>/blob/main/notebooks/03_bridging.ipynb) |
| 04 — Disease Modules (optional) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<username>/<repo>/blob/main/notebooks/04_disease_modules_optional.ipynb) |

## Data setup

`data/` and `networks/` need to actually be in this repo for Step 0 of Notebook 1 to find them (they're currently gitignored — see below). Once you've confirmed what's safe to include, remove the relevant lines from `.gitignore` and push the CSVs.

**Before running anything yourself:** open `notebooks/01_matching_and_loading.ipynb` and set `GITHUB_REPO = "<username>/<repo>"` to your actual repo path — that's the one line every notebook depends on to find your data.

## For the organizer

`prep_scripts/` contains two one-time setup scripts, run **before** the workshop, not by attendees:

- `build_metabolite_gene_bridge.py` → produces `metabolite_gene_bridge.csv`
- `build_human_metabolite_network.py` → produces `metabolite_network.csv`

Both write files that belong in the `networks/` folder the notebooks expect.

## Repo structure

```
notebooks/        the four workshop notebooks
prep_scripts/      organizer-only, run once ahead of time
data/               (DEGs.csv, DE_proteins.csv, DE_metabolites.csv - if made public)
networks/           (ppi_network.csv, transcriptome_network.csv, metabolite_network.csv, metabolite_gene_bridge.csv - if made public)
```
