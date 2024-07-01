# Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.14403)
[![GitHub Stars](https://img.shields.io/github/stars/matciotola/hyperspectral_pansharpening_toolbox?style=social)](https://github.com/matciotola/hyperspectral_pansharpening_toolbox)
![Visitors Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fhits.dwyl.com%2Fmatciotola%2Fhyperspectral_pansharpening_toolbox.json&style=flat&color=blue)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/68906119170e489cbf98512fac6b9571)](https://app.codacy.com/gh/matciotola/hyperspectral_pansharpening_toolbox/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)


[Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives](https://github.com/matciotola/hyperspectral_pansharpening_toolbox) ([ArXiv](https://github.com/matciotola/hyperspectral_pansharpening_toolbox)): In this paper, 
state-of-the-art methods for hyperspectral pansharpening, 
comprising both model- and deep learning-based ones, 
are reviewed, reimplemented/trained in a shared framework, and critically compared on a benchmark dataset designed to test the methods under challenging conditions. 
Compared to traditional multispectral pansharpening,
for which the well-known Wald's protocol has been a polar reference for the related community
both for solutions design and for quality assessment,
the hyperspectral case presents additional problems to be properly taken into account.
First, 
the much larger number of bands poses non-trivial computational issues, 
especially for data demanding techniques such as deep learning ones.
Second, 
the hyperspectral imaging processes are characterized by high noise levels and acquisition errors
that impact on the quality of the fused products.
Finally, the spectral coverage of the panchromatic image spans from the visible spectrum to the near infrared,
leaving out the most of the hyperspectral range.
This makes hard to infer about the spatial details in those uncovered bands 
for which the spatial guidance of the panchromatic image is unreliable.
In addition,
deep learning solutions, 
nowadays the most promising research line,
are highly data-dependent which makes their assessment very hard. 

In light of the above considerations,
this work moves from the roots, 
designing a sufficiently large and rich dataset
for the purposes of training and/or validation and test
of all compared methods.
Then, all selected solutions, representative of the state-of-the-art,
have been reimplemented, and optimized where needed, 
in a unique PyTorch framework.
Finally, a thorough critical comparative analysis has been carried out,
leveraging on the most credited quality indicators and criteria for the given task.
The experimental analysis allowed to 
highlight the main limits of the given methods in terms of spectral and/or spatial quality and of computational efficiency.
Besides,
to ensure full reproducibility of the results and pave a solid and reliable road for 
future researches on this topic,
the framework 
(codes for methods and assessment, and links to retrieve the dataset)
as a unique Python-based benchmark toolbox.

## Cite HS Toolbox
If you use this toolbox in your research, please use the following BibTeX entry.

    @article{
     ** AVAILABLE SOON **
     }


## License

Copyright (c) 2024 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved.
This software should be used, reproduced and modified only for informational and nonprofit purposes.

By downloading and/or using any of these files, you implicitly agree to all the
terms of the license, as specified in the document [`LICENSE`](https://github.com/matciotola/Lambda-PNN/LICENSE.txt)
(included in this package)

## Prerequisites

All the functions and scripts were tested on Windows and Ubuntu O.S., with these constrains:

*   Python 3.10.10
*   PyTorch 2.0.0
*   Cuda 11.7 or 11.8 (For GPU acceleration).

the operation is not guaranteed with other configurations.

## Installation

*   Install [Anaconda](https://www.anaconda.com/products/individual) and [git](https://git-scm.com/downloads)
*   Create a folder in which save the algorithm
*   Download the algorithm and unzip it into the folder or, alternatively, from CLI:

<!---->

    git clone https://github.com/matciotola/hyperspectral_pansharpening_toolbox

*   Create the virtual environment with the `hs_pan_toolbox_environment.yml`

<!---->

    conda env create -n hs_pan_toolbox_environment -f hs_pan_toolbox_environment.yml

*   Activate the Conda Environment

<!---->

    conda activate hs_pan_toolbox_environment

* Edit the 'preambol.yaml' file with the correct paths and the desired algorithms to run

*   Test it

<!---->

    python main.py 



