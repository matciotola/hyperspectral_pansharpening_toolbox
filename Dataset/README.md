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

To have access to PRISMAs data, please follow the instructions at the following link: [PRISMA User Registration](https://prismauserregistration.asi.it/)


## Download PRISMA Images

Once you have registered to the PRISMA portal, you can download the images from the ASI server. You can download the images at the following link: [PRISMA Data](https://prisma.asi.it/)
The list of images used in this work is available in [`list_images.csv`](https://github.com/matciotola/hyperspectral_pansharpening_toolbox/tree/main/Dataset/list_images.csv) file.
The steps for downloading the images are summarized in the following guide:


<object data="https://github.com/matciotola/hyperspectral_pansharpening_toolbox/blob/main/Dataset/HowTo_download_PRISMA_images.pdf" type="application/pdf" width="100%">
    <embed src="https://github.com/matciotola/hyperspectral_pansharpening_toolbox/blob/main/Dataset/HowTo_download_PRISMA_images.pdf" type="application/pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/matciotola/hyperspectral_pansharpening_toolbox/blob/main/Dataset/HowTo_download_PRISMA_images.pdf">Download PDF</a>.</p>
    </embed>
</object>



    