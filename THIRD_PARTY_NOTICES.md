# Third-Party Notices

This extension is based on, and depends on, the following third-party components.

- **SlicerUpperAirwaySegmentator** (upstream): Apache-2.0.
  https://github.com/capenaka/SlicerUpperAirwaySegmentator
  Upstream nnU-Net model and Slicer extension AUAE is based on.
- **nnU-Net v2**: Apache-2.0. https://github.com/MIC-DKFZ/nnUNet
  Isensee et al., Nature Methods 2021.
- **SlicerNNUNet** (`NNUNet` Slicer extension): provides `SlicerNNUNetLib` used for inference.
- **3D Slicer**: BSD-style Slicer license. https://slicer.org
- **PyGithub**, **SciPy**, **NumPy**, **light-the-torch**, **PyTorch**: respective OSS licenses.

The bundled model weights are downloaded at runtime from the upstream project's GitHub
releases and remain subject to the upstream project's terms.

## Attribution and citation

The upstream segmentation model and extension are the work of the University of Alberta
authors. If you use this extension, cite:

- Gianoni-Capenakas S, Matos A, Dot G, Schouman T, Chaurasia A, Pliska B, Lagravere M,
  Panithakumar K. Segmentation of the Upper Airway using Deep learning - nnUNet. Journal of
  Dentistry, 2026. https://doi.org/10.1016/j.jdent.2026.106507
- Isensee F, et al. nnU-Net. Nature Methods, 2021. https://doi.org/10.1038/s41592-020-01008-z

Upstream repository: https://github.com/capenaka/SlicerUpperAirwaySegmentator
