cmake_minimum_required(VERSION 3.13.4)

project(AUAE)

#-----------------------------------------------------------------------------
# Extension meta-information
# Based on capenaka/SlicerUpperAirwaySegmentator (Apache-2.0). Upstream
# authorship is preserved; the maintainers and additions are appended.
set(EXTENSION_HOMEPAGE "https://github.com/pietro98vr/SlicerAUAE")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Alejandro Matos Camarillo (uAlberta), Silvia Capenakas (uAlberta), Manuel Lagravere (uAlberta), Pietro Montagna (University of Verona), Fabio Lonardi (University of Verona)")
set(EXTENSION_DESCRIPTION "Based on UpperAirwaySegmentator: fully automatic upper-airway CT/CBCT nnU-Net segmentation, plus airway extension for flow modelling, JSON-driven batch processing with per-volume output subfolders, mutually-exclusive island cleanup, and an autonomous CUDA-aware dependency preflight. The embedded Segment Editor is removed; upstream STL/OBJ/NIFTI export is kept.")
set(EXTENSION_DEPENDS "NNUNet") # NNUNet transitively installs the PyTorch extension (CUDA torch)

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AUAE)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
