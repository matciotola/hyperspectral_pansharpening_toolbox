# Q2n benchmark: recursive PyTorch vs. pyfastcd hypercomplex products

Q2n multiplies pixels as Cayley-Dickson numbers whose dimension is the number of bands rounded up to a power of two: 256 for the 159 bands of PRISMA. The toolbox ships a recursive PyTorch implementation of that product and uses the compiled kernels of the optional [pyfastcd](https://pypi.org/project/pyfastcd/) package when it is installed. `q2n_fastcd.py` checks that the two give the same results and measures the difference in speed.

## Running it

From the root of the repository, in the toolbox environment:

```
pip install "pyfastcd>=1.4.0"
python benchmarks/q2n_fastcd.py            # full run, about 25 minutes
python benchmarks/q2n_fastcd.py --quick    # small images, about a minute
python benchmarks/q2n_fastcd.py --json results.json   # also save the raw timings
```

The script prints Markdown tables like the ones below. It needs no PRISMA product: the images are seeded random data with the band counts and sizes of the test set. Before timing anything it compares the products of the two backends for every dimension from 2 to 256 and stops if they differ by more than `1e-12`. It also compares the Q2n index the two backends return for every image.

## What is measured

- **Products of one block.** One pixelwise product of a 32x32 block plus one product of the block means: the hypercomplex arithmetic `q_index_metric` performs per block, and the only code that differs between the backends.
- **Complete Q2n evaluations.** `Q2n()(outputs, labels)` on float64 CPU tensors under `torch.no_grad()`, as `Metrics/evaluation.py` calls it from `main.py`. The two PRISMA rows use the sizes of the test set: `evaluation_fr` applies Q2n to the 200x200 low-resolution image, and `evaluation_rr` to the 600x600 ground truth after its border cut, which leaves 579x579. Rounds alternate the backends, so background load affects both alike, and the table gives medians.
- **Ceiling.** The same evaluations with a product that costs nothing. Everything else in Q2n (per-band normalisation, means, slicing) is identical for both backends, so this is the largest speedup that any replacement of the products could reach.

## Reference results

Measured on 2026-11-13 on a shared host in a single process.

- cpu: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
- platform: Linux-7.2.11-arch1-1-x86_64-with-glibc2.44
- python: 3.14.8
- torch: 2.15.0+cpu
- numpy: 2.5.4
- pyfastcd: 1.4.0 (manylinux wheel from PyPI)
- threads: 40

Largest relative difference between the backends' products: 5.1e-16. Largest difference in Q2n: 0.0e+00.

### Hypercomplex products of one 32x32 block (milliseconds)

| N | torch | pyfastcd | speedup |
| ---: | ---: | ---: | ---: |
| 4 | 0.436 | 0.084 | 5.2x |
| 8 | 1.877 | 0.098 | 19.2x |
| 16 | 7.439 | 0.179 | 41.6x |
| 32 | 29.650 | 0.386 | 76.8x |
| 64 | 121.032 | 0.845 | 143.2x |
| 128 | 514.452 | 1.864 | 276.0x |
| 256 | 2045.227 | 4.658 | 439.1x |

### Complete Q2n evaluations (seconds)

This table reports the median [min, max] time for a full Q2n evaluation per method over interleaved rounds. "wins" counts the rounds in which pyfastcd was faster. "ceiling" is the speedup with products that cost nothing.

| image | torch | pyfastcd | speedup | wins | ceiling | Q2n (both backends) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 bands, 256x256 | 0.193 [0.187, 0.199] | 0.080 [0.078, 0.083] | 2.4x | 7/7 | 2.9x | 0.985182 |
| 32 bands, 256x256 | 2.484 [2.469, 2.527] | 0.297 [0.291, 0.306] | 8.4x | 5/5 | 10.0x | 0.985228 |
| 102 bands, 200x200 | 24.998 [24.935, 25.742] | 0.803 [0.800, 0.842] | 31.1x | 3/3 | 38.3x | 0.985228 |
| 159 bands (PRISMA), 200x200, as in evaluation_fr | 100.238 [99.321, 118.346] | 1.627 [1.488, 1.925] | 61.6x | 3/3 | 81.0x | 0.985217 |
| 159 bands (PRISMA), 579x579, as in evaluation_rr | 913.676 [913.676, 913.676] | 10.936 [10.936, 10.936] | 83.5x | 1/1 | 103.3x | 0.985225 |

The last row is a single round because one evaluation with the recursive backend takes about a quarter of an hour. A separate run of that case on the same machine took 801.7 s and 12.0 s (67x), so some variance may be observed.

## Reading the results

The recursion makes about N²/4 leaf calls per product, each a handful of small tensor operations, so its time grows fourfold with every doubling of the number of bands and reaches two seconds per block at N=256. The compiled kernel does the same arithmetic in a few milliseconds using a novel O(N log N) algorithm. A complete Q2n evaluation gains less than the products alone, because the per-band normalisation of every block is shared code that this change does not touch; with pyfastcd that normalisation is most of what remains, as the "ceiling" column shows.

For the test set of the toolbox (four full-resolution and four reduced-resolution PRISMA images) one algorithm's Q2n evaluations add up to roughly an hour with the recursive backend and under a minute with pyfastcd, going by the two PRISMA rows above.

The pyfastcd backend runs on the CPU. Tensors on another device are copied to the CPU and back, which was not timed here. Products of tensors that require gradients keep using the recursion.
