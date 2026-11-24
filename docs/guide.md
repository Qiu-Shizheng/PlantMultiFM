# PlantMultiFM user guide

## Inputs and feature matching

Use a cells-by-features matrix of raw integer RNA counts or binary ATAC accessibility. Supply gene identifiers or assembly-matched peak names in column order. Use one panel per assay axis, not a union of zero-padded panels. A measured zero and an unmeasured feature have different meanings. For ATAC fragment counts, validate that counts are nonnegative raw counts before converting detected peaks to one.

Download a panel and `SHA256SUMS` from the [release](https://github.com/Qiu-Shizheng/PlantMultiFM/releases/tag/v0.1.0). Check its SHA-256 digest against the corresponding entry. Panel names are available from `panel.names["rna"]` and `panel.names["atac"]`. Align names exactly. Resolve genome builds, gene aliases and duplicate features before inference. A different peak set requires an explicit coordinate mapping on the same assembly.

## Use an AnnData object

```python
import anndata as ad   # optional: pip install anndata
from plantmultifm import Panel

p = Panel("arabidopsis.npz")
a = ad.read_h5ad("plant_rna.h5ad")
assert a.var_names.is_unique
common = a.var_names.intersection(p.names["rna"], sort=False)
print(f"{len(common)} of {a.n_vars} measured genes match the panel")
counts = a.layers["counts"] if "counts" in a.layers else a.X  # must be raw
z = p.encode(counts[:, a.var_names.get_indexer(common)], common, "rna")
a.obsm["X_plantmultifm"] = z
peak_scores = p.predict(z, p.names["atac"][:100], "rna_to_atac")
```

Rows retain input cell order. The output columns follow the requested target order. The encoder processes cells in batches (default 128). Request genes or peaks in blocks when the full prediction matrix is large. A small overlap changes the information available to the model: report the number of matched features with each result.

## Tasks

| Task | Encoder input | Output and use |
|---|---|---|
| `atac_to_rna` | ATAC | Nonnegative native expression score. Rank nuclei **within a gene** with this score to reproduce the transcript-detection endpoint. |
| `atac_to_rna_detection` | ATAC | Sigmoid output from the separate binary detection head. |
| `rna_to_atac` | RNA | Sigmoid accessibility score per requested region. |
| `masked_rna_detection` | RNA with target genes removed | Sigmoid score from the RNA self-reconstruction head. Remove target columns before encoding, including from depth and support. |

`encode` returns a cell state, not a cell-type label. For supervised annotation, fit a classifier to labelled reference states and apply it to query states from the same encoder. Fit any scaling or dimensionality reduction on the reference. Sigmoid scores are model outputs, not calibrated experimental probabilities.

## Command line

Use one feature name per line, without a header. The feature file describes the matrix columns. The target file specifies the outputs.

```bash
plantmultifm embed --panel rice.npz --modality rna --matrix counts.npz \
  --features measured_genes.txt --output rna_states.npy
plantmultifm predict --panel rice.npz --modality atac --matrix binary_peaks.npz \
  --features peaks.txt --task atac_to_rna --targets genes.txt --output scores.npy
```

Dense inputs use NumPy `.npy`; sparse inputs use SciPy `sparse.save_npz`. No cell labels, target-assay depth or network connection are required for inference.

