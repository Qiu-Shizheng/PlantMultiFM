"""Command-line access to panel feature lists, embeddings and molecular scores."""
import argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from .model import Panel


def lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def main():
    p = argparse.ArgumentParser(prog="plantmultifm")
    subs = p.add_subparsers(dest="command", required=True)
    for name in ("features", "embed", "predict"):
        a = subs.add_parser(name)
        a.add_argument("--panel", required=True)
        a.add_argument("--modality", choices=["rna", "atac"], required=True)
        a.add_argument("--output", required=True)
        if name != "features":
            a.add_argument("--matrix", required=True, help="Cells x features, SciPy CSR .npz or NumPy .npy")
            a.add_argument("--features", required=True, help="Feature names in matrix-column order")
        if name == "predict":
            a.add_argument("--task", choices=["atac_to_rna","atac_to_rna_detection","rna_to_atac","masked_rna_detection"], required=True)
            a.add_argument("--targets", required=True)
    a = p.parse_args()
    panel = Panel(a.panel)
    if a.command == "features":
        Path(a.output).write_text("\n".join(panel.names[a.modality])+"\n", encoding="utf-8")
        return
    feats = lines(a.features)
    if a.command == "predict":
        targets = lines(a.targets)
        if (a.task.startswith("atac_to_") and a.modality != "atac") or (not a.task.startswith("atac_to_") and a.modality != "rna"):
            p.error("Task and input modality do not match.")
        if a.task == "masked_rna_detection" and set(feats).intersection(targets):
            p.error("Remove all target genes from the input feature axis before encoding.")
    matrix = sparse.load_npz(a.matrix) if a.matrix.endswith(".npz") else np.load(a.matrix, allow_pickle=False)
    z = panel.encode(matrix, feats, a.modality)
    with open(a.output, "wb") as out:
        np.save(out, panel.predict(z, targets, a.task) if a.command == "predict" else z, allow_pickle=False)


if __name__ == "__main__":
    main()

