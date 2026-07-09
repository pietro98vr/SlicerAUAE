# NOTICES

## Derivation

This project is based on **SlicerUpperAirwaySegmentator**
(https://github.com/capenaka/SlicerUpperAirwaySegmentator), licensed under the
**Apache License 2.0** (see `LICENSE.md`). The upstream extension and its nnU-Net model
provide the automatic upper-airway segmentation core and the STL/OBJ/NIFTI export that this
project keeps unchanged.

Upstream authors: Alejandro Matos Camarillo, Silvia Capenakas, Manuel Lagravere
(University of Alberta).

## Authors

- Dr. Pietro Montagna, DDS, MSc, PhD Student, pietro.montagna@univr.it
- Dr. Fabio Lonardi, MD, OMFS, PhD Student, fabio.lonardi@univr.it

Head and Neck Department, Department of Surgery, Dentistry, Pediatrics and Gynecology,
University of Verona, Verona, Italy.

## Modifications (contributed under Apache-2.0)

- Airway-extension step for flow modelling: `AUAELib/AirwayExtension.py`.
- Batch processing with a JSON template and per-volume output subfolders:
  `AUAELib/BatchProcessor.py`, `Resources/batch_template.json`.
- Mutually-exclusive island cleanup (remove-small vs keep-largest), reimplemented in NumPy/SciPy.
- Autonomous, CUDA-aware dependency preflight: `AUAELib/Dependencies.py`.
- Removed the embedded Segment Editor.

## Citation

Please cite the upstream paper (Gianoni-Capenakas et al., Journal of Dentistry 2026) and
nnU-Net (Isensee et al., Nature Methods 2021). See `README.md` and `CITATION.cff`.

## Third-party components

See `THIRD_PARTY_NOTICES.md`.
