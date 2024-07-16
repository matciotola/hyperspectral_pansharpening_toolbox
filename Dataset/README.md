# Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives - Dataset

A major contribution of this work is to provide a dataset that meets the following requirements:

- it should include many acquisitions, with as much diversity as possible in terms of geographical regions, land covers, acquisition conditions;
- training and test sets should not share the same acquisitions;
- both the training (especially) and the test set should include much more than a single image;
- all images should share the same set of spectral bands for cross-image validation and testing and to allow fair comparison between all methods, regardless of their ability to handle a variable number of bands;
- images should be truly multi-resolution, with a high-resolution PAN associated with the HS image, to enable real-world full-resolution testing;
- the dataset should be freely available to the community to ensure its widest use and reproducibility of results.


Considering the restrictive data-sharing policies widespread in the remote sensing field, we decided to use the PRISMA (PRecursore IperSpettrale della Missione Applicativa) images,
shared on-demand by the Italian Space Agency (ASI) for research purposes only. Due to ASI Data Policy, we cannot share the images directly. However, we provide this `README.md` to download the images from the ASI server and a script to elaborate them correctly.

## Register to PRISMA Portal

The delivery of the PRISMA Products is subject to the User registration on the PRISMA portal. The
registration procedure shall require the User to:
1. Register himself/herself by providing its personal data;
2. Declare the Purposes of Use of the ORIGINAL Product (i.e. Pansharpening);
3. Fill in the Project Form, including the personal data of any possible Contractor and/or Consultant and/or
Affiliated User and/or Commercial Team User, where applicable.

Following the registration procedures, a specific account will be assigned to the User. All personal data
provided by the User shall be electronically stored by ASI, by the ASI contractors that implement the
operational activities of the PRISMA system in the name and on behalf of ASI, including the user support; all personal data provided by the User shall be processed by the relevant national competent authorities. By
requesting the ORIGINAL Products, the Main User acknowledges the characteristics of the PRISMA
Products and the restrictions to which their provision and use may be subject to, as specified in these General
Conditions and in the attached License to Use. (Source: [PRISMA Portal](https://prisma.asi.it/))

To have access to PRISMA data, please follow the instructions at the following link: [PRISMA User Registration](https://prismauserregistration.asi.it/)


## Download PRISMA Images

Once you have registered to the PRISMA portal, you can download the images from the ASI server. You can download the images at the following link: [PRISMA Data](https://prisma.asi.it/)
The list of images used in this work is available in [`list_images.csv`](https://github.com/matciotola/hyperspectral_pansharpening_toolbox/tree/main/Dataset/list_images.csv) file.
The steps for downloading the images are summarized in this [**GUIDE**](https://github.com/matciotola/hyperspectral_pansharpening_toolbox/blob/main/Dataset/HowTo_download_PRISMA_images.pdf):



## Elaborate PRISMA Images

To elaborate the PRISMA images correctly, unzip all the files in a specif folder. 
Then, you can use the script [`elaborate_PRISMA_images.py`](https://github.com/matciotola/hyperspectral_pansharpening_toolbox/blob/main/elaborate_PRISMA_images.py) available in this repository to produce the final dataset.
<!---->

    python elaborate_PRISMA_images.py -i /path/to/folder/with/unzipped/files -o /path/to/output/folder

If no output folder is provided, the script will save the dataset in the Dataset folder of the toolbox.

If everything is done correctly, you have a dataset folder with the following structure:

```
Dataset
    ├──  Training
    │       ├──  Full_Resolution
    │       │       ├──  PRS_L2D_STD_20200627102358_20200627102402_0001_01.mat
    │       │       ├──  PRS_L2D_STD_20200627102358_20200627102402_0001_02.mat
    │       │       ├──  ...
    │       │       └──  PRS_L2D_STD_20231120165641_20231120165646_0001_16.mat
    │       ├──  Reduced_Resolution
    │       │       ├──  PRS_L2D_STD_20200627102358_20200627102402_0001_01.mat
    │       │       ├──  PRS_L2D_STD_20200627102358_20200627102402_0001_02.mat
    │       │       ├──  ...
    │       │       └──  PRS_L2D_STD_20231120165641_20231120165646_0001_16.mat
    ├──  Validation
    │       ├──  Full_Resolution
    │       │       ├──  PRS_L2D_STD_20200627102358_20200627102402_0001_01.mat
    │       │       ├──  PRS_L2D_STD_20200627102358_20200627102402_0001_02.mat
    │       │       ├──  ...
    │       │       └──  PRS_L2D_STD_20231120165641_20231120165646_0001_02.mat
    │       ├──  Reduced_Resolution
    │       │       ├──  PRS_L2D_STD_20200627102358_20200627102402_0001_01.mat
    │       │       ├──  PRS_L2D_STD_20200627102358_20200627102402_0001_02.mat
    │       │       ├──  ...
    │       │       └──  PRS_L2D_STD_20231120165641_20231120165646_0001_02.mat
    ├──  Test
    │       ├──  Full_Resolution
    │       │       ├──  PRS_L2D_STD_20220905101901_20220905101905_0001_CAGLIARI_FR.mat
    │       │       ├──  PRS_L2D_STD_20230824100356_20230824100400_0001_UDINE_FR.mat
    │       │       ├──  PRS_L2D_STD_20230908173127_20230908173131_0001_KANSAS_FR.mat
    │       │       └──  PRS_L2D_STD_20231120102229_20231120102233_0001_TABASCO_FR.mat
    │       ├──  Reduced_Resolution
    │       │       ├──  PRS_L2D_STD_20220905101901_20220905101905_0001_CAGLIARI_FR.mat
    │       │       ├──  PRS_L2D_STD_20230824100356_20230824100400_0001_UDINE_FR.mat
    │       │       ├──  PRS_L2D_STD_20230908173127_20230908173131_0001_KANSAS_FR.mat
    │       │       └──  PRS_L2D_STD_20231120102229_20231120102233_0001_TABASCO_FR.mat
    
```

## Problems or Questions

If you have any problems or questions, please contact me by email ([matteo.ciotola@unina.it](mailto:matteo.ciotola@unina.it?subject=[Hyperspectral Toolbox]PRISMA Dataset Issues)).