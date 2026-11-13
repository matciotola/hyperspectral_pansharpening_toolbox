"""Hypercomplex (Cayley-Dickson) products used by the Q2n index.

A pixel with N = 2**n bands is treated as an element of the 2**n-dimensional
Cayley-Dickson algebra (complex numbers for N=2, quaternions for N=4,
octonions for N=8, ...), stored as its N real coefficients.  PRISMA images
have 159 bands, which Q2n pads to N=256.

Two interchangeable backends compute the product:

* ``torch``: the recursive implementation ported from the MATLAB
  pansharpening toolbox.  Always available.
* ``pyfastcd``: the compiled kernels of the optional ``pyfastcd`` package
  (``pip install "pyfastcd>=1.4.0"``).  Used automatically when it is
  installed.

Both backends return the same values.  Writing ``x . y`` for the standard
Cayley-Dickson product

    (a, b)(c, d) = (ac - conj(d) b, da + b conj(c)),

which is the convention ``pyfastcd`` implements, the toolbox recursion
computes

    x * y = T(conj(x . y)),

where ``T`` is the parity automorphism ``e_i -> (-1)**popcount(i) e_i``.
``T o conj`` only changes the sign of some coordinates: coordinate ``i`` keeps
its sign when ``i == 0`` or ``popcount(i)`` is odd and is negated otherwise.
The ``pyfastcd`` backend therefore multiplies with the compiled kernel and
applies that fixed sign pattern.  Q2n uses only the norm of these products,
which the sign pattern does not change, but applying it keeps the two
backends equal coordinate by coordinate.

The recursion makes about N**2 / 4 leaf calls, each a handful of tensor
operations, so its cost is dominated by call overhead: roughly one second
per product at N=256 on a CPU, largely independent of the block size.  The
compiled kernel runs on the CPU in float64; tensors on another device are
copied to the CPU and the result is moved back.  The recursion is still used
when gradients are required, because the compiled kernel is not part of the
autograd graph, and for N < 4, where the product is a single tensor expression.
"""
from functools import lru_cache

import numpy as np
import torch

try:
    import pyfastcd
except ImportError:
    pyfastcd = None

BACKEND = 'torch' if pyfastcd is None else 'pyfastcd'


def cayley_dickson_property_1d_torch(onion1, onion2):
    _, N = onion1.size()

    if N > 1:
        L = int(N / 2)
        a = onion1[:, :L]
        b = onion1[:, L:]
        neg = - torch.ones(b.shape, dtype=b.dtype, device=b.device)
        neg[:, 0] = 1
        b = b * neg
        c = onion2[:, :L]
        d = onion2[:, L:]
        d = d * neg

        if N == 2:
            ris = torch.cat(((a * c) - (d * b), (a * d) + (c * b)), dim=1)
        else:
            ris1 = cayley_dickson_property_1d_torch(a, c)
            ris2 = cayley_dickson_property_1d_torch(d, torch.cat((torch.unsqueeze(b[:, 0], 1), -b[:, 1:]), dim=1))
            ris3 = cayley_dickson_property_1d_torch(torch.cat((torch.unsqueeze(a[:, 0], 1), -a[:, 1:]), dim=1), d)
            ris4 = cayley_dickson_property_1d_torch(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.cat((aux1, aux2), dim=1)
    else:
        ris = onion1 * onion2

    return ris


def cayley_dickson_property_2d_torch(onion1, onion2):
    _, dim3, _, _ = onion1.size()
    if dim3 > 1:
        L = int(dim3 / 2)

        a = onion1[:, 0:L, :, :]
        b = onion1[:, L:, :, :]
        neg = - torch.ones(b.shape, dtype=b.dtype, device=b.device)
        neg[:, 0, :, :] = 1
        b = b * neg
        c = onion2[:, 0:L, :, :]
        d = onion2[:, L:, :, :]
        d = d * neg
        if dim3 == 2:
            ris = torch.cat(((a * c) - (d * b), (a * d) + (c * b)), dim=1)
        else:
            ris1 = cayley_dickson_property_2d_torch(a, c)
            ris2 = cayley_dickson_property_2d_torch(d, b * neg)
            ris3 = cayley_dickson_property_2d_torch(a * neg, d)
            ris4 = cayley_dickson_property_2d_torch(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = torch.cat((aux1, aux2), dim=1)
    else:
        ris = onion1 * onion2

    return ris


@lru_cache(maxsize=None)
def _fastcd_context(n):
    """The pyfastcd algebra of dimension ``n`` and the sign pattern of ``T o conj``."""
    sign = np.array([1.0 if i == 0 or bin(i).count('1') % 2 else -1.0 for i in range(n)])
    sign.flags.writeable = False
    return pyfastcd.Algebra(dimension=n), sign


def _fastcd_product(x, y):
    """Product of two float64 arrays whose last axis holds the coefficients."""
    algebra, sign = _fastcd_context(x.shape[-1])
    out = np.empty(x.shape, dtype=np.float64)
    algebra.multiply(algebra.asarray(x), algebra.asarray(y), out=out)
    out *= sign
    return out


def cayley_dickson_property_1d_fastcd(onion1, onion2):
    """Product of two (batch, N) tensors of hypercomplex numbers."""
    if onion1.shape[1] == 1:
        return onion1 * onion2
    x = np.ascontiguousarray(onion1.detach().cpu().numpy(), dtype=np.float64)
    y = np.ascontiguousarray(onion2.detach().cpu().numpy(), dtype=np.float64)
    ris = torch.from_numpy(_fastcd_product(x, y))
    return ris.to(device=onion1.device, dtype=onion1.dtype)


def cayley_dickson_property_2d_fastcd(onion1, onion2):
    """Pixelwise product of two (batch, N, H, W) tensors of hypercomplex numbers."""
    if onion1.shape[1] == 1:
        return onion1 * onion2
    # pyfastcd expects the coefficients on the last axis
    x = np.ascontiguousarray(onion1.detach().cpu().numpy().transpose(0, 2, 3, 1), dtype=np.float64)
    y = np.ascontiguousarray(onion2.detach().cpu().numpy().transpose(0, 2, 3, 1), dtype=np.float64)
    ris = torch.from_numpy(_fastcd_product(x, y)).permute(0, 3, 1, 2)
    return ris.to(device=onion1.device, dtype=onion1.dtype)


def _use_fastcd(onion1, onion2):
    n = onion1.shape[1]
    if pyfastcd is None or n < 4 or n & (n - 1):
        return False
    return not (torch.is_grad_enabled() and (onion1.requires_grad or onion2.requires_grad))


def cayley_dickson_property_1d(onion1, onion2):
    if _use_fastcd(onion1, onion2):
        return cayley_dickson_property_1d_fastcd(onion1, onion2)
    return cayley_dickson_property_1d_torch(onion1, onion2)


def cayley_dickson_property_2d(onion1, onion2):
    if _use_fastcd(onion1, onion2):
        return cayley_dickson_property_2d_fastcd(onion1, onion2)
    return cayley_dickson_property_2d_torch(onion1, onion2)
