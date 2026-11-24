import os
import unittest
import numpy as np
from scipy import sparse
from plantmultifm import Panel


@unittest.skipUnless(os.getenv("PLANTMULTIFM_PANEL"), "Set PLANTMULTIFM_PANEL to a downloaded panel")
class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Panel(os.environ["PLANTMULTIFM_PANEL"])

    def test_axis_and_batch_invariance(self):
        for m in ("rna", "atac"):
            axis = self.p.names[m][:4]
            x = np.array([[1, 1, 0, 0], [0, 0, 1, 0]])
            a = self.p.encode(x, axis, m)
            b = self.p.encode(sparse.csr_matrix(x[:, ::-1]), axis[::-1], m, batch_size=1)
            np.testing.assert_allclose(a, b, atol=2e-5)
            self.assertEqual(a.shape, (2, 256))

    def test_scores_and_input_validation(self):
        for m, task, tgt in [("rna","rna_to_atac","atac"),("atac","atac_to_rna","rna"),
                             ("rna","masked_rna_detection","rna"),("atac","atac_to_rna_detection","rna")]:
            axis = self.p.names[m][:2]
            z = self.p.encode([[1, 0]], axis, m)
            p = self.p.predict(z, self.p.names[tgt][4:7], task)
            self.assertEqual(p.shape, (1, 3))
            self.assertTrue(np.isfinite(p).all() and (p >= 0).all())
            if task != "atac_to_rna":
                self.assertTrue((p <= 1).all())
        for x, axis, m in [([[2]],self.p.names["atac"][:1],"atac"), ([[0.5]], self.p.names["rna"][:1],"rna"),
                           ([[1]],["absent_feature"],"rna"), ([[1, 1]],["a", "a"],"rna")]:
            with self.assertRaises(ValueError):
                self.p.encode(x, axis, m)

