import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hmm import GaussianHMM  


def toy_model() -> GaussianHMM:
    """2 states, 1-D Gaussian emissions centred at 0 and 3."""
    m = GaussianHMM(2, var_floor=1e-6)
    m.startprob_ = np.array([0.6, 0.4])
    m.transmat_ = np.array([[0.7, 0.3], [0.2, 0.8]])
    m.means_ = np.array([[0.0], [3.0]])
    m.vars_ = np.array([[1.0], [0.5]])
    return m


def gauss(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


class TestToyHMM(unittest.TestCase):
    X = np.array([[0.1], [2.7], [3.2], [-0.4], [0.3]])

    def joint(self, m, path):
        """p(path, X) computed directly from the model definition."""
        p = m.startprob_[path[0]] * gauss(self.X[0, 0], m.means_[path[0], 0], m.vars_[path[0], 0])
        for t in range(1, len(path)):
            p *= m.transmat_[path[t - 1], path[t]] * gauss(self.X[t, 0], m.means_[path[t], 0], m.vars_[path[t], 0])
        return p

    def test_forward_matches_brute_force(self):
        m = toy_model()
        paths = list(itertools.product([0, 1], repeat=len(self.X)))
        brute = np.log(sum(self.joint(m, p) for p in paths))
        self.assertAlmostEqual(m.score(self.X), brute, places=10)

    def test_viterbi_matches_brute_force(self):
        m = toy_model()
        paths = list(itertools.product([0, 1], repeat=len(self.X)))
        best = max(paths, key=lambda p: self.joint(m, p))
        path, logp = m.viterbi(self.X)
        self.assertEqual(tuple(path), best)
        self.assertAlmostEqual(logp, np.log(self.joint(m, best)), places=10)
        self.assertEqual(tuple(path), (0, 1, 1, 0, 0))

    def test_posteriors_sum_to_one(self):
        g = toy_model().posteriors(self.X)
        np.testing.assert_allclose(g.sum(axis=1), 1.0)

    def test_baum_welch_does_not_decrease_likelihood(self):
        rng = np.random.default_rng(0)
        m = toy_model()
        seqs = [rng.normal(size=(30, 1)) + 3 * (rng.random((30, 1)) > 0.5) for _ in range(3)]
        m.tol = -np.inf  
        m.max_iter = 15
        m.fit_baum_welch(seqs)
        self.assertTrue(np.all(np.diff(m.history_) > -1e-8))


if __name__ == "__main__":
    unittest.main()
