# Third-party notices

Project-specific training and packaging code follows the Apache-2.0 license in
`LICENSE`. The vendored model code retains its original file headers and licenses;
it is not relicensed by the project-level license.

## NVIDIA semantic segmentation

The HRNet/OCR multi-scale attention model and supporting code are derived from
[NVIDIA semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation),
Copyright 2020 NVIDIA Corporation, BSD-3-Clause. The copied license is in
`licenses/NVIDIA-BSD-3-Clause.txt`.

This repository packages the exact local source family used for the baseline.
Its handoff recorded source commit `dc274b3281026fef3a1cb395df70d552df784dfd`;
that identifier has not been independently matched to a public upstream revision.
`docs/source_provenance.json` records per-file hashes of the actual source snapshot.

Local adaptations: package-relative imports, native PyTorch interpolation AMP
compatibility instead of Apex, logging instead of runx, only the HRNet backbone
factory, removal of unused upstream training CLI/configuration, and the existing
`np.int` to `int` compatibility change. Parameter names and numerical model
operations are preserved; see the extraction validation report.

## Microsoft HRNet / OCR

`_vendor/network/hrnetv2.py` and `ocr_utils.py` retain Microsoft MIT headers and
credit Ke Sun and Jingyi Xie. Source identified in those files:
[HRNet-Semantic-Segmentation, HRNet-OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR).
The license is retained in `licenses/HRNet-MIT.txt`. No upstream syncbn extension
is included; this package uses PyTorch BatchNorm2d.

## Data and weights

Mapillary Vistas images and annotations are not included. Selected model weights,
including the original65-class pretrained checkpoint and original taxonomy, are
retained in `models/` via Git LFS with byte-for-byte source hashes. Their own
terms apply separately from the project code license; this addition does not
relicense the upstream weights or the dataset. See `models/README.md`.
Private vehicle images, calibration, records and account credentials must not be
added to this repository. Class IDs and evaluation split IDs document the task;
they do not include dataset pixels.
