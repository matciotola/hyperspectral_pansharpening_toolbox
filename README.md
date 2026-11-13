# Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2407.01355)
[![GitHub Stars](https://img.shields.io/github/stars/matciotola/hyperspectral_pansharpening_toolbox?style=social)](https://github.com/matciotola/hyperspectral_pansharpening_toolbox)
![Visitors Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fhits.dwyl.com%2Fmatciotola%2Fhyperspectral_pansharpening_toolbox.json&style=flat&color=blue)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/68906119170e489cbf98512fac6b9571)](https://app.codacy.com/gh/matciotola/hyperspectral_pansharpening_toolbox/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


[Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives](https://ieeexplore.ieee.org/document/10804644) ([ArXiv](https://arxiv.org/abs/2407.01355)): Hyperspectral pansharpening consists of fusing a high-resolution panchromatic band and a low-resolution hyperspectral image to obtain a new image with high resolution in both the spatial and spectral domains.
These remote sensing products are valuable for a wide range of applications, driving ever growing research efforts.
Nonetheless, results still do not meet application demands.
In part, this comes from the technical complexity of the task: compared to multispectral pansharpening, many more bands are involved, in a spectral range only partially covered by the panchromatic component and with overwhelming noise.
However, another major limiting factor is the absence of a comprehensive framework for the rapid development and accurate evaluation of new methods.
This paper attempts to address this issue.
 
We started by designing a dataset large and diverse enough to allow reliable training (for data-driven methods) and testing of new methods.
Then, we selected a set of state-of-the-art methods, following different approaches, characterized by promising performance, and reimplemented them in a single PyTorch framework.
Finally, we carried out a critical comparative analysis of all methods,  using the most accredited quality indicators.
The analysis highlights the main limitations of current solutions in terms of spectral/spatial quality and computational efficiency, and suggests promising research directions.
 
To ensure full reproducibility of the results and support future research,
the framework (including codes, evaluation procedures and links to the dataset) is shared as a single Python-based reference benchmark toolbox.

## Cite HS Toolbox
If you use this toolbox in your research, please use the following BibTeX entry.

    @ARTICLE{Ciotola2024HSPansharp,
      author={Ciotola, Matteo and Guarino, Giuseppe and Vivone, Gemine and Poggi, Giovanni and Chanussot, Jocelyn and Plaza, Antonio and Scarpa, Giuseppe},
      journal={IEEE Geoscience and Remote Sensing Magazine}, 
      title={Hyperspectral Pansharpening: Critical review, tools, and future perspectives}, 
      year={2024},
      volume={},
      number={},
      pages={2-29},
      keywords={Pansharpening;Spatial resolution;Reviews;Training;Benchmark testing;Accuracy;Sensors;Proposals;Optimization;Limiting},
      doi={10.1109/MGRS.2024.3509139}
    }



## License

Copyright (c) 2024 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/matciotola/hyperspectral_pansharpening_toolbox/LICENSE.md)
(included in this package)

## Dataset

Considering the restrictive data-sharing policies widespread in the remote sensing field, we decided to use the PRISMA (PRecursore IperSpettrale della Missione Applicativa) images,
shared on-demand by the Italian Space Agency (ASI) for research purposes only. Due to ASI Data Policy, we cannot share the images directly.
However, we provide extensive instructions on how to download and elaborate the images correctly in the [`Dataset`](https://github.com/matciotola/hyperspectral_pansharpening_toolbox/tree/main/Dataset) folder of this repository.
For any problem or question, please contact me at If you have any problems or questions, please contact me by email ([matteo.ciotola@unina.it](mailto:matteo.ciotola@unina.it)).


## Prerequisites

All the functions and scripts were tested on Windows and Ubuntu O.S., with these constrains:

*   Python 3.10.10
*   PyTorch >= 2.0.0
*   Cuda  11.8 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

*   Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads)
*   Create a folder in which save the toolbox
*   Download the toolbox and unzip it into the folder or, alternatively, from CLI:

<!---->

    git clone https://github.com/matciotola/hyperspectral_pansharpening_toolbox

*   Create the virtual environment with the `hs_pan_toolbox_environment.yml`

<!---->
    # For Windows/Linux users
    conda env create -n hs_pan_toolbox_environment -f hs_pan_toolbox_environment.yml

    # For MacOS users
    conda env create -n hs_pan_toolbox_environment -f hs_pan_toolbox_environment_mac.yml 

*   Activate the Conda Environment

<!---->

    conda activate hs_pan_toolbox_environment

* Edit the 'preambol.yaml' file with the correct paths and the desired algorithms to run

*   Test it

<!---->

    python main.py 

## Faster Q2n with pyfastcd (optional)

The Q2n index multiplies pixels as Cayley-Dickson numbers, of dimension 256 for the 159 bands of PRISMA. When the [pyfastcd](https://pypi.org/project/pyfastcd/) package is installed, these products are computed by its compiled kernels instead of the recursive PyTorch implementation, with the same results: on a PRISMA-sized test image a Q2n evaluation takes seconds instead of minutes (see [`benchmarks`](benchmarks/README.md)). The package is listed in the environment files; in an existing environment install it with

    pip install "pyfastcd>=1.4.0"

Without it the toolbox works as before. The checks comparing the two implementations are in [`tests`](tests) and run with

    python -m unittest discover -s tests -v
