"""Tests for the hypercomplex products behind Q2n and their two backends.

Run from the root of the repository:

    python -m unittest discover -s tests -v

The tests that compare the backends are skipped when pyfastcd is not installed.
"""
import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Metrics import hypercomplex as hc  # noqa: E402
from Metrics import metrics_rr as rr  # noqa: E402

HAVE_FASTCD = hc.pyfastcd is not None
BACKENDS = {'torch': (hc.cayley_dickson_property_1d_torch, hc.cayley_dickson_property_2d_torch)}
if HAVE_FASTCD:
    BACKENDS['pyfastcd'] = (hc.cayley_dickson_property_1d_fastcd, hc.cayley_dickson_property_2d_fastcd)


def randn(*shape, seed=0, dtype=torch.float64):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def conj(x):
    return torch.cat((x[:, :1], -x[:, 1:]), dim=1)


def q2n_with(backend, outputs, labels):
    saved = rr.cayley_dickson_property_1d, rr.cayley_dickson_property_2d
    rr.cayley_dickson_property_1d, rr.cayley_dickson_property_2d = BACKENDS[backend]
    try:
        with torch.no_grad():
            index, index_map = rr.Q2n()(outputs.clone(), labels.clone())
        return index.item(), index_map
    finally:
        rr.cayley_dickson_property_1d, rr.cayley_dickson_property_2d = saved


def image_pair(bands, side, seed):
    g = torch.Generator().manual_seed(seed)
    labels = torch.rand(1, bands, side, side, generator=g, dtype=torch.float64) * 4000
    outputs = labels + torch.randn(1, bands, side, side, generator=g, dtype=torch.float64) * 200
    return outputs, labels


class TestProducts(unittest.TestCase):
    def test_selected_backend(self):
        self.assertEqual(hc.BACKEND, 'pyfastcd' if HAVE_FASTCD else 'torch')
        self.assertIs(rr.cayley_dickson_property_1d, hc.cayley_dickson_property_1d)
        self.assertIs(rr.cayley_dickson_property_2d, hc.cayley_dickson_property_2d)

    def test_algebra_properties(self):
        for name, (mult, mult2d) in BACKENDS.items():
            with self.subTest(backend=name):
                # N=2 is ordinary complex multiplication
                x, y = randn(1, 2, seed=1), randn(1, 2, seed=2)
                z = complex(*x[0].tolist()) * complex(*y[0].tolist())
                self.assertTrue(torch.allclose(mult(x, y), torch.tensor([[z.real, z.imag]], dtype=torch.float64)))
                for n in (1, 2, 4, 8, 16):
                    x, y = randn(3, n, seed=n), randn(3, n, seed=n + 100)
                    # x * conj(x) = |x|^2 for every N
                    expected = torch.zeros(3, n, dtype=torch.float64)
                    expected[:, 0] = (x ** 2).sum(dim=1)
                    self.assertTrue(torch.allclose(mult(x, conj(x)), expected))
                    # |x * y| = |x||y| up to the octonions
                    if n <= 8:
                        self.assertTrue(torch.allclose(mult(x, y).norm(dim=1), x.norm(dim=1) * y.norm(dim=1)))
                    # the pixelwise product is the single product applied to every pixel
                    X, Y = randn(2, n, 3, 5, seed=n + 200), randn(2, n, 3, 5, seed=n + 300)
                    Z = mult2d(X, Y)
                    self.assertEqual(Z.shape, X.shape)
                    self.assertTrue(torch.allclose(Z[:, :, 2, 4], mult(X[:, :, 2, 4], Y[:, :, 2, 4])))

    @unittest.skipUnless(HAVE_FASTCD, 'pyfastcd is not installed')
    def test_product_parity(self):
        for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            with self.subTest(N=n):
                X, Y = randn(2, n, 5, 7, seed=n), randn(2, n, 5, 7, seed=n + 1000)
                self.assertTrue(torch.allclose(hc.cayley_dickson_property_2d_fastcd(X, Y),
                                               hc.cayley_dickson_property_2d_torch(X, Y), rtol=1e-12, atol=1e-12))
                x, y = X[:, :, 0, 0].clone(), Y[:, :, 0, 0].clone()
                self.assertTrue(torch.allclose(hc.cayley_dickson_property_1d_fastcd(x, y),
                                               hc.cayley_dickson_property_1d_torch(x, y), rtol=1e-12, atol=1e-12))

    @unittest.skipUnless(HAVE_FASTCD, 'pyfastcd is not installed')
    def test_dtype_and_memory_layout(self):
        # float32 tensors come back as float32, equal to the recursion up to float32 rounding
        X, Y = randn(1, 16, 6, 6, seed=5, dtype=torch.float32), randn(1, 16, 6, 6, seed=6, dtype=torch.float32)
        Z = hc.cayley_dickson_property_2d_fastcd(X, Y)
        self.assertEqual(Z.dtype, torch.float32)
        self.assertTrue(torch.allclose(Z, hc.cayley_dickson_property_2d_torch(X, Y), rtol=1e-4, atol=1e-4))
        z = hc.cayley_dickson_property_1d_fastcd(X[:, :, 0, 0], Y[:, :, 0, 0])
        self.assertEqual(z.dtype, torch.float32)
        # non-contiguous inputs
        X, Y = randn(2, 32, 8, 8, seed=7)[:, ::2, ::2, 1::2], randn(2, 32, 8, 8, seed=8)[:, ::2, ::2, 1::2]
        self.assertFalse(X.is_contiguous())
        self.assertTrue(torch.allclose(hc.cayley_dickson_property_2d_fastcd(X, Y),
                                       hc.cayley_dickson_property_2d_torch(X, Y), rtol=1e-12, atol=1e-12))
        # the inputs are left untouched
        X, Y = randn(1, 8, 4, 4, seed=9), randn(1, 8, 4, 4, seed=10)
        X0, Y0 = X.clone(), Y.clone()
        hc.cayley_dickson_property_2d_fastcd(X, Y)
        self.assertTrue(torch.equal(X, X0) and torch.equal(Y, Y0))

    @unittest.skipUnless(HAVE_FASTCD, 'pyfastcd is not installed')
    def test_dispatch(self):
        def calls(X, Y):
            with mock.patch.object(hc, 'cayley_dickson_property_2d_fastcd',
                                   wraps=hc.cayley_dickson_property_2d_fastcd) as fast:
                Z = hc.cayley_dickson_property_2d(X, Y)
            return fast.call_count, Z

        X, Y = randn(1, 8, 4, 4, seed=1), randn(1, 8, 4, 4, seed=2)
        self.assertEqual(calls(X, Y)[0], 1)
        # below the quaternions the product is a single tensor expression
        self.assertEqual(calls(X[:, :2], Y[:, :2])[0], 0)
        # a band count that is not a power of two is left to the recursion
        self.assertEqual(calls(X[:, :6], Y[:, :6])[0], 0)
        # tensors that need gradients stay in the autograd graph
        Xg = X.clone().requires_grad_()
        count, Z = calls(Xg, Y)
        self.assertEqual(count, 0)
        self.assertTrue(Z.requires_grad)
        Z.sum().backward()
        self.assertEqual(Xg.grad.shape, Xg.shape)
        with torch.no_grad():
            self.assertEqual(calls(Xg, Y)[0], 1)
        # the 1d product follows the same rules
        with mock.patch.object(hc, 'cayley_dickson_property_1d_fastcd',
                               wraps=hc.cayley_dickson_property_1d_fastcd) as fast:
            hc.cayley_dickson_property_1d(X[:, :, 0, 0], Y[:, :, 0, 0])
            hc.cayley_dickson_property_1d(Xg[:, :, 0, 0], Y[:, :, 0, 0])
        self.assertEqual(fast.call_count, 1)


class TestQ2n(unittest.TestCase):
    def test_identical_images(self):
        for name in BACKENDS:
            with self.subTest(backend=name):
                _, labels = image_pair(5, 64, seed=1)
                index, index_map = q2n_with(name, torch.round(labels), torch.round(labels))
                self.assertAlmostEqual(index, 1.0, places=5)
                self.assertEqual(tuple(index_map.shape), (1, 2, 2))

    @unittest.skipUnless(HAVE_FASTCD, 'pyfastcd is not installed')
    def test_q2n_parity(self):
        # 5 bands padded to 8 with spatial padding, 20 bands padded to 32,
        # and one block with the 159 bands of PRISMA padded to 256
        for bands, side in ((5, 70), (20, 64), (159, 32)):
            with self.subTest(bands=bands, side=side):
                outputs, labels = image_pair(bands, side, seed=bands)
                fast, fast_map = q2n_with('pyfastcd', outputs, labels)
                slow, slow_map = q2n_with('torch', outputs, labels)
                self.assertTrue(0 < slow < 1)
                self.assertAlmostEqual(fast, slow, places=6)
                self.assertTrue(torch.allclose(fast_map, slow_map, rtol=1e-6, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
