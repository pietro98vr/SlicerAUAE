# NOTICES

## Derivation

This project is based on **SlicerUpperAirwaySegmentator**
(https://github.com/capenaka/SlicerUpperAirwaySegmentator), licensed under the
**Apache License 2.0** (see `LICENSE.md`). The upstream extension and its nnU-Net model
provide the automatic upper-airway segmentation core and the mesh and labelmap export that
this project builds on.

Upstream authors: Alejandro Matos Camarillo, Silvia Capenakas, Manuel Lagravere
(University of Alberta).

## Authors

- Dr. Pietro Montagna, DDS, MSc, PhD Student, pietro.montagna@univr.it
- Dr. Fabio Lonardi, MD, OMFS, PhD Student, fabio.lonardi@univr.it

Head and Neck Department, Department of Surgery, Dentistry, Pediatrics and Gynecology,
University of Verona, Verona, Italy.

## Modifications (contributed under Apache-2.0)

- Post-processing in NumPy/SciPy (`AUAELib/AirwayExtension.py`): island cleanup (remove-small
  or keep-largest), recovery of the paranasal sinuses, optional external face-air segment, and
  an inferior airway extension for flow modelling.
- Three-step panel (segment, refine in the embedded Segment Editor, extend) and export to
  STL, OBJ, NIFTI or NRRD, per segment or merged: `AUAELib/SegmentationWidget.py`.
- Batch processing of a folder (volumes and DICOM series) or a JSON template, with per-case
  output subfolders: `AUAELib/BatchProcessor.py`, `Resources/batch_template.json`.
- Headless command-line batch runner: `CLI/auae_batch.py`.
- MCP server to run the extension from an assistant: `MCP/auae_mcp.py`.
- Autonomous, CUDA-aware dependency preflight: `AUAELib/Dependencies.py`.
- Removed the unused upstream `Signal.py` helper.

## Citation

Please cite the upstream paper (Gianoni-Capenakas et al., Journal of Dentistry 2026) and
nnU-Net (Isensee et al., Nature Methods 2021). See `README.md`.

## Third-party components

See `THIRD_PARTY_NOTICES.md`.
