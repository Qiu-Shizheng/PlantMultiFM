"""RNA and chromatin inference with a frozen, feature-indexed panel."""
import numpy as np
from scipy import sparse
from scipy.special import ndtr, expit


def _labels(values):
    a = np.asarray(values, dtype=str)
    if a.ndim != 1 or not len(a) or len(set(a)) != len(a) or np.any(a == ""):
        raise ValueError("Expected nonempty, unique feature names.")
    return a


def _gelu(x):
    return x * ndtr(x)


class Panel:
    """Load a public .npz panel. Feature names must match its gene or peak axis."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as f:
            if int(f["schema"]) != 1:
                raise ValueError("Unsupported panel schema.")
            self.name = str(f["panel"])
            self.checkpoint = str(f["checkpoint"])
            self.names = {m: _labels(f[m + "_names"]) for m in ("rna", "atac")}
            self._w = {k: f[k] for k in f.files if "." in k or k.endswith(("_input", "_output"))}
        self._index = {m: {n: i for i, n in enumerate(a)} for m, a in self.names.items()}
        for m, axis in self.names.items():
            for suffix in ("input", "output"):
                if self._w[m + "_" + suffix].shape != (len(axis), 256):
                    raise ValueError("Mismatched molecular basis.")

    def _ids(self, modality, features):
        if modality not in self.names:
            raise ValueError("Modality must be rna or atac.")
        labels = _labels(features)
        try:
            return np.array([self._index[modality][x] for x in labels])
        except KeyError as e:
            raise ValueError(f"Feature absent from {self.name} {modality}: {e.args[0]}") from e

    def _linear(self, x, name):
        y = x @ self._w[name + ".weight"].T
        bias = self._w.get(name + ".bias")
        return y if bias is None else y + bias

    def encode(self, counts, features, modality, batch_size=128):
        """Encode a cells-by-measured-features matrix. Omit unmeasured columns."""
        idx = self._ids(modality, features)
        x = sparse.csr_matrix(counts, copy=True)
        if (x.ndim != 2 or x.shape[1] != len(idx) or not x.shape[0]
                or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("Invalid matrix shape or batch size.")
        if x.dtype.kind not in "buif" or not np.isfinite(x.data).all() or np.any(x.data < 0) or np.any(x.data != np.floor(x.data)):
            raise ValueError("Expected finite nonnegative integer counts.")
        x = x.astype(np.float32)
        x.sum_duplicates()
        x.eliminate_zeros()
        if modality == "atac" and np.any(x.data != 1):
            raise ValueError("ATAC input must be binary.")
        basis = self._w[modality + "_input"][idx]
        states = []
        for start in range(0, x.shape[0], batch_size):
            row = x[start:start + batch_size]
            depth = np.asarray(row.sum(axis=1)).ravel()
            n = np.diff(row.indptr)
            cov = np.log1p(np.column_stack((depth, n, np.full(len(n), len(idx))))).astype("f") / 10
            val = row.multiply((10000 / np.maximum(depth, 1))[:, None]).tocsr()
            np.log1p(val.data, out=val.data)
            present = row.copy()
            present.data.fill(1)
            z = np.column_stack(((val @ basis) / np.float32(np.sqrt(len(idx))),
                                 (present @ basis) / np.float32(np.sqrt(len(idx))), cov))
            p = "encoders." + modality + "."
            z = self._linear(_gelu(self._linear(z, p + "0")), p + "2")
            z = (z - z.mean(1, keepdims=True)) / np.sqrt(z.var(1, keepdims=True) + 1e-5)
            states.append(z * self._w[p + "3.weight"] + self._w[p + "3.bias"])
        return np.concatenate(states)

    def predict(self, states, targets, task):
        """Return cells-by-target scores for a named source-to-target task."""
        modes = {"atac_to_rna": ("rna", "cell.rna.0", "query.rna", "prior.rna"),
                 "atac_to_rna_detection": ("rna", "cell.rna.0", "detection_query", "detection_prior"),
                 "rna_to_atac": ("atac", "cell.atac.0", "query.atac", "prior.atac"),
                 "masked_rna_detection": ("rna", "self_rna.cell.0", "self_rna.detection_query", "self_rna.detection_prior")}
        if task not in modes:
            raise ValueError("Unknown task: " + str(task))
        mod, cell, q, b = modes[task]
        states = np.asarray(states, dtype="f")
        if states.ndim != 2 or not len(states) or states.shape[1] != 256 or not np.isfinite(states).all():
            raise ValueError("Expected finite cells-by-256 states.")
        basis = self._w[mod + "_output"][self._ids(mod, targets)]
        logits = _gelu(self._linear(states, cell)) @ self._linear(basis, q).T / 16 + self._linear(basis, b).T
        return np.logaddexp(0, logits) if task == "atac_to_rna" else expit(logits)

