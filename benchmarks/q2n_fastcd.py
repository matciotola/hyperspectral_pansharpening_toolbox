#!/usr/bin/env python
"""Benchmark Q2n with the torch (recursive) and pyfastcd hypercomplex backends.

Run from the root of the repository, with pyfastcd installed:

    pip install "pyfastcd>=1.4.0"
    python benchmarks/q2n_fastcd.py            # full run, about 25 minutes
    python benchmarks/q2n_fastcd.py --quick    # small images, about a minute
    python benchmarks/q2n_fastcd.py --json results.json

The script
1. checks that both backends give the same products, and refuses to time
   anything if they do not;
2. times the hypercomplex products of one block (the only code that differs
   between the backends);
3. times complete Q2n evaluations, called as Metrics/evaluation.py calls
   them, and checks that both backends return the same index.  Rounds
   alternate the backends so that background load affects both alike, and
   medians are reported.  A third variant whose products cost nothing gives
   the largest speedup any replacement of the products could reach, because
   the rest of Q2n is shared.

The images are seeded random data with the band counts and sizes of the
toolbox's test set; no PRISMA product is needed.
"""
import argparse
import json
import os
import platform
import sys
import time
import timeit

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from Metrics import hypercomplex as hc  # noqa: E402
from Metrics import metrics_rr as rr  # noqa: E402

_ZEROS = {}


def _free(onion1, onion2):
    """Simulates a product that costs (almost) nothing."""
    z = _ZEROS.get(onion1.shape)
    if z is None:
        z = _ZEROS[onion1.shape] = torch.zeros(onion1.shape, dtype=onion1.dtype)
    return z


def backends():
    return {
        'torch': (hc.cayley_dickson_property_1d_torch, hc.cayley_dickson_property_2d_torch),
        'pyfastcd': (hc.cayley_dickson_property_1d_fastcd, hc.cayley_dickson_property_2d_fastcd),
        'free': (_free, _free),
    }


def q2n_with(backend, outputs, labels):
    saved = rr.cayley_dickson_property_1d, rr.cayley_dickson_property_2d
    rr.cayley_dickson_property_1d, rr.cayley_dickson_property_2d = backends()[backend]
    try:
        with torch.no_grad():
            start = time.perf_counter()
            value = rr.Q2n()(outputs.clone(), labels.clone())[0].item()
            return time.perf_counter() - start, value
    finally:
        rr.cayley_dickson_property_1d, rr.cayley_dickson_property_2d = saved


def best_time(f, target=0.05, repeat=5):
    """Best seconds per call, with enough calls per repeat to last about `target`."""
    start = time.perf_counter(); f(); once = time.perf_counter() - start
    number = max(1, int(target / max(once, 1e-9)))
    return min(timeit.repeat(f, number=number, repeat=repeat)) / number


def cpu_name():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except (IOError, OSError):
        pass
    return platform.processor() or platform.machine()


def environment():
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'pyfastcd': hc.pyfastcd.__version__,
        'threads': torch.get_num_threads(),
        'platform': platform.platform(),
        'cpu': cpu_name(),
    }


def cases(quick, rng):
    # (description, bands, side, rounds).  159 bands is PRISMA; after the border
    # cut of evaluation_rr the 600x600 reduced-resolution test images are 579x579,
    # and evaluation_fr applies Q2n to the 200x200 low-resolution images.
    if quick:
        spec = (('8 bands, 128x128', 8, 128, 3),
                ('32 bands, 128x128', 32, 128, 3),
                ('159 bands (PRISMA), 64x64', 159, 64, 2))
    else:
        spec = (('8 bands, 256x256', 8, 256, 7),
                ('32 bands, 256x256', 32, 256, 5),
                ('102 bands, 200x200', 102, 200, 3),
                ('159 bands (PRISMA), 200x200, as in evaluation_fr', 159, 200, 3),
                ('159 bands (PRISMA), 579x579, as in evaluation_rr', 159, 579, 1))
    out = []
    for name, bands, side, rounds in spec:
        labels = rng.uniform(0, 4000, (1, bands, side, side))
        outputs = labels + rng.normal(0, 200, labels.shape)
        out.append((name, torch.from_numpy(outputs), torch.from_numpy(labels), rounds))
    return out


def check_parity(dims, rng):
    worst = 0.0
    for N in dims:
        X = torch.from_numpy(rng.standard_normal((2, N, 8, 8)))
        Y = torch.from_numpy(rng.standard_normal((2, N, 8, 8)))
        ref = hc.cayley_dickson_property_2d_torch(X, Y)
        worst = max(worst, ((hc.cayley_dickson_property_2d_fastcd(X, Y) - ref).abs().max() / ref.abs().max()).item())
        x, y = X[:, :, 0, 0].clone(), Y[:, :, 0, 0].clone()
        ref = hc.cayley_dickson_property_1d_torch(x, y)
        worst = max(worst, ((hc.cayley_dickson_property_1d_fastcd(x, y) - ref).abs().max() / ref.abs().max()).item())
    if worst > 1e-12:
        sys.exit('Backends disagree (products: %.1e relative); not timing.' % worst)
    return worst


def per_block(size, dims, rng):
    rows = []
    for N in dims:
        X = torch.from_numpy(rng.standard_normal((1, N, size, size)))
        Y = torch.from_numpy(rng.standard_normal((1, N, size, size)))
        x, y = X.mean(dim=(2, 3)), Y.mean(dim=(2, 3))
        row = {'N': N}
        for name in ('torch', 'pyfastcd'):
            mult, mult2d = backends()[name]
            row[name] = best_time(lambda: (mult2d(X, Y), mult(x, y))) * 1e3
        rows.append(row)
    return rows


def end_to_end(all_cases):
    rows = []
    for name, outputs, labels, rounds in all_cases:
        times = {'torch': [], 'pyfastcd': [], 'free': []}
        values = {}
        for _ in range(rounds):
            for backend in times:
                seconds, values[backend] = q2n_with(backend, outputs, labels)
                times[backend].append(seconds)
        rows.append({'case': name, 'rounds': rounds, 'seconds': times,
                     'q2n': {k: values[k] for k in ('torch', 'pyfastcd')}})
        print('  timed %s' % name, file=sys.stderr)
    return rows


def report(env, parity, block_rows, e2e_rows, size):
    print('## Environment\n')
    for key in ('cpu', 'platform', 'python', 'torch', 'numpy', 'pyfastcd', 'threads'):
        print('- %s: %s' % (key, env[key]))
    q2n_diff = max(abs(r['q2n']['torch'] - r['q2n']['pyfastcd']) for r in e2e_rows)
    print('\n## Parity\n')
    print('Largest relative difference between the backends\' products: %.1e. '
          'Largest difference in Q2n: %.1e.' % (parity, q2n_diff))
    print('\n## Hypercomplex products of one %dx%d block (milliseconds)\n' % (size, size))
    print('One pixelwise product of the block plus one product of the block means, '
          'as in `q_index_metric`. Best of 5 repeats.\n')
    print('| N | torch | pyfastcd | speedup |')
    print('| ---: | ---: | ---: | ---: |')
    for r in block_rows:
        print('| %d | %.3f | %.3f | %.1fx |' % (r['N'], r['torch'], r['pyfastcd'], r['torch'] / r['pyfastcd']))
    print('\n## Complete Q2n evaluations (seconds)\n')
    print('Median [min, max] over interleaved rounds. "wins" counts the rounds in which '
          'pyfastcd was faster. "ceiling" is the speedup with products that cost nothing.\n')
    print('| image | torch | pyfastcd | speedup | wins | ceiling | Q2n (both backends) |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for r in e2e_rows:
        py, fc, fr = (np.array(r['seconds'][k]) for k in ('torch', 'pyfastcd', 'free'))
        fmt = lambda t: '%.3f [%.3f, %.3f]' % (np.median(t), t.min(), t.max())
        mp, mf, m0 = np.median(py), np.median(fc), np.median(fr)
        print('| %s | %s | %s | %.1fx | %d/%d | %.1fx | %.6f |' % (
            r['case'], fmt(py), fmt(fc), mp / mf, int((fc < py).sum()), r['rounds'], mp / m0, r['q2n']['pyfastcd']))
    return q2n_diff


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--quick', action='store_true', help='small images and fewer rounds')
    parser.add_argument('--json', metavar='PATH', help='also write raw timings to PATH')
    parser.add_argument('--size', type=int, default=32, help='block size for the per-block timing (default 32, the Q2n default)')
    parser.add_argument('--threads', type=int, default=0, help='number of torch threads (default: torch\'s own default)')
    args = parser.parse_args()
    if hc.pyfastcd is None:
        sys.exit('pyfastcd is not installed: pip install "pyfastcd>=1.4.0"')
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    rng = np.random.default_rng(7)
    dims = (4, 8, 16, 32, 64, 128, 256)
    all_cases = cases(args.quick, rng)
    env = environment()
    parity = check_parity((2,) + dims, rng)
    block_rows = per_block(args.size, dims, rng)
    e2e_rows = end_to_end(all_cases)
    q2n_diff = report(env, parity, block_rows, e2e_rows, args.size)
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'environment': env, 'quick': args.quick,
                       'parity': {'products_relative': parity, 'q2n_absolute': q2n_diff},
                       'per_block_ms': block_rows, 'end_to_end': e2e_rows}, f, indent=1)
    if q2n_diff > 1e-6:
        sys.exit('Backends disagree on Q2n by %.1e.' % q2n_diff)


if __name__ == '__main__':
    main()
