# Selected baseline

Architecture: HRNet-W48 + OCR, 0.5/1.0 multi-scale attention, direct eight classes.
Selected update: 1500 from the 4000-update direct-eight-class training run.

Checkpoint SHA256:
`fbcbc1312837b8e8e31c3d4c9521aab3c5bb44be8c28360a142db29f2c313ffc`

The initial checkpoint for this run was a trained 124-class HRNet-OCR (step10000):
`410ddff9467ad41401e26f23e9273a8333f154f80a4211b006d535cb608223fd`.
It is required to repeat the 124→8 initialization route; neither checkpoint is
included or automatically downloaded. Strictly load the selected eight-class
checkpoint for deployment; never repeat head initialization on that checkpoint.

## Recorded evaluation, percentages

These are historical measured results, not a fresh full validation of the
extracted repository. Evaluation uses BF16 and a maximum image side of1536.
Deployment image reviews use native-resolution FP32.

| Same570-image panel | Selected baseline | A: lower LR, +250 | B: lower LR + reduced weights, +750 |
| --- | ---: | ---: | ---: |
| Lane IoU | 63.356 | 61.810 | 64.456 |
| Exclude IoU | 60.163 | 60.069 | 62.065 |
| Lane/exclude mean IoU | 61.760 | 60.939 | 63.261 |
| Lane pixels misclassified as exclude | 1.497 | 1.253 | 0.859 |
| Flat-surface pixels misclassified as lane/exclude | 1.232 | 1.439 | 0.936 |
| Arrow filtering recall | 90.608 | 90.577 | 87.090 |
| Hatching filtering recall | 80.541 | 79.575 | 74.619 |

Full2000-image lane/exclude mean IoU: 59.602 / 58.883 / 61.151.
B improves precision but sacrifices arrow/hatching recall; it did not pass the
predeclared maximum1-percentage-point recall loss. The baseline remains selected.
A/B each ran1000 extra updates from the same selected checkpoint. A changes LR
relative to the original training recipe; there was no original-LR continuation
control. B versus A isolates the lane/exclude loss weights4→2.

All419 validation images containing arrows are in the570 panel; the remaining1430
have no arrow ground truth and are not a new independent test set. Hongqi vehicle
images were reviewed without ground-truth annotations. Neither these metrics nor
successful numerical parity establish map usability or cross-platform robustness.
