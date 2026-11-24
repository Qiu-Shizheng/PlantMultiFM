# PlantMultiFM

Predict transcript detection from chromatin accessibility, predict accessibility from RNA, or embed individual plant cells in 256 dimensions. This Python package runs the frozen PlantMultiFM encoder and molecular readouts on CPU using NumPy and SciPy.

[User guide](docs/guide.md) | [中文指南](docs/guide.zh.md) | [Pretrained panels and checksums](https://github.com/Qiu-Shizheng/PlantMultiFM/releases/tag/v0.1.0)

## Install and run

```bash
python -m pip install "git+https://github.com/Qiu-Shizheng/PlantMultiFM.git@v0.1.0"
curl -fLO https://github.com/Qiu-Shizheng/PlantMultiFM/releases/download/v0.1.0/arabidopsis.npz
plantmultifm features --panel arabidopsis.npz --modality rna --output genes.txt
```

```python
from plantmultifm import Panel
import numpy as np

panel = Panel("arabidopsis.npz")
# Small synthetic input to check installation: rows are nuclei, columns are peaks.
peaks = panel.names["atac"][:4]
x = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
states = panel.encode(x, peaks, "atac")
scores = panel.predict(states, panel.names["rna"][:3], "atac_to_rna")
print(states.shape, scores.shape)  # (2, 256), (2, 3)
```

For experimental data, supply all measured and matched source features, including columns with zero counts. A panel stores the learned molecular bases and readout weights on the indicated assay axes. Each was derived from the same pretrained model.

| Panel | Public assay | Genes | Regions |
|---|---|---:|---:|
| Arabidopsis | [GSE235510, Case 1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE235510) | 27,641 | 24,489 |
| Rice | [GSE232863, multi-organ atlas](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE232863) | 35,782 | 82,204 |
| Camptotheca | [PRJNA1243444, paired nuclei](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1243444) | 37,347 | 48,455 |

The package supports inference on the listed feature axes. The original studies provide the experimental count matrices. See the guide for sparse inputs, feature alignment, score interpretation and masked-gene prediction.

