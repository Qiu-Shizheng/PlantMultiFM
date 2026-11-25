"""Embed public root cells and score masked transcripts from the remaining RNA."""
import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from plantmultifm import Panel

FILES = {
    "matrix": "1c4b70c0718a3ac227430c16b9cd8d2741c79fdf85ec4d0825c64b7266df6da2",
    "metadata": "643f1af7b683c191604aef5b97bf71eb7a390149938143e52e6e86476f1af886",
}
TARGETS = ["AT1G52070", "AT2G43610", "AT5G10130", "AT5G60530"]


def run(panel_path, matrix, metadata, out):
    for key, path in [("matrix", matrix), ("metadata", metadata)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != FILES[key]:
            raise ValueError(f"Unexpected {key} file. Use the GEO files in the guide.")
    panel = Panel(panel_path)
    if panel.name != "GSE235510_Case1":
        raise ValueError("Use the Arabidopsis panel.")
    obs = pd.read_csv(metadata, index_col="Cell")
    labels = obs.loc[obs["set"] == "I", "cell_type_simple"].dropna()
    labels = labels[~labels.str.lower().isin(["not significant", "unknown", "unassigned"])]
    columns = pd.read_csv(matrix, sep="\t", nrows=0, index_col=0).columns
    available = np.array(sorted(c for c in columns if c + "_I" in labels.index))
    cells = np.sort(np.random.default_rng(42).choice(available, min(512, len(available)), replace=False))
    if not len(cells):
        raise ValueError("The author metadata did not match any matrix barcodes.")
    groups = labels.loc[[c + "_I" for c in cells]].to_numpy()
    blocks, names = [], []
    axis = set(panel.names["rna"])
    for chunk in pd.read_csv(matrix, sep="\t", index_col=0, chunksize=512):
        keep = chunk.index.isin(axis)
        blocks.append(sparse.csr_matrix(chunk.loc[keep, cells].to_numpy()))
        names.extend(chunk.index[keep])
    x = sparse.vstack(blocks).T.tocsr()
    genes = pd.Index(names)
    if not genes.is_unique or not set(TARGETS).issubset(genes):
        raise ValueError("Gene axis mismatch.")
    hidden = genes.isin(TARGETS)
    z = panel.encode(x, genes, "rna")
    masked = panel.encode(x[:, ~hidden], genes[~hidden], "rna")
    scores = panel.predict(masked, TARGETS, "masked_rna_detection")
    detected = x[:, genes.get_indexer(TARGETS)].toarray() > 0
    if not np.isfinite(scores).all() or np.any(np.ptp(scores, axis=0) == 0):
        raise ValueError("Invalid predictions.")
    xy = PCA(n_components=2, svd_solver="full").fit_transform(z)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(cell=[c + "_I" for c in cells], annotation=groups,
                      PC1=xy[:,0], PC2=xy[:,1])).to_csv(out/"cells.csv", index=False)
    fig, (a,b) = plt.subplots(1,2,figsize=(9.2,4.5), layout="constrained")
    for j, label in enumerate(sorted(set(groups))):
        v = xy[groups == label]
        a.scatter(*v.T, s=10, alpha=.75, color=plt.get_cmap("tab10")(j), label=label, lw=0)
    a.set(xlabel="PC1", ylabel="PC2", title="Root cell states")
    a.legend(loc="upper center", bbox_to_anchor=(.5,-.17), ncol=3, frameon=False, fontsize=8)
    for j,gene in enumerate(TARGETS):
        order = np.argsort(scores[:,j], kind="stable")
        b.plot(range(1,6), [detected[ix,j].mean() for ix in np.array_split(order,5)],
               marker="o", ms=4, label=gene)
    b.set(xlabel="Predicted rank quintile within gene", ylabel="Fraction with measured RNA",
          xticks=range(1,6), ylim=(0,1), title="Detection of withheld transcripts")
    b.legend(frameon=False, fontsize=8)
    for ax in (a,b):
        ax.spines[["right","top"]].set_visible(False)
    fig.savefig(out/"root_atlas.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "matrix", "metadata", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    run(a.panel, a.matrix, a.metadata, a.out)
