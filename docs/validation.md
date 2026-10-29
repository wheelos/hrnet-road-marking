# Extraction validation

This report concerns packaging and execution correctness, not a new model-quality
claim. The selected checkpoint and existing training results are unchanged.

Completed checks on2026-10-29:

- Strictly load the selected eight-class step1500 checkpoint: all1903 state entries.
- Native RGB inputs1920×1084 and1600×900: packaged PyTorch logits exactly equal the
  previously recorded original implementation (maximum difference0, hard labels100%).
- First8 actual training crops (ordinary and complete-arrow sampling): input tensors
  and mapped labels exactly equal the original training tools.
- CPU regression checks cover taxonomy, palette-index decoding, ignored labels,
  deterministic sampler suffix, full-arrow crops, neighboring-instance separation,
  metric denominators, RGB preprocessing and eight-channel model forward.

- Real training smoke check: three optimizer updates, four512×768 microbatches
  per update, two validation images. A second run stopped at update1 and resumed
  to update3. Sampling, final CPU/CUDA RNG states and optimizer parameter groups
  match the uninterrupted run; losses and gradients are finite. CUDA/BF16
  backward is not bitwise deterministic: maximum parameter difference was
  3.81e-6. This is execution/resume verification, not additional model training
  for deployment. The selected baseline file was not modified.
- ONNX export/checker passed. Native1920×1084 and1600×900 ONNX Runtime CPU outputs
  compared with the previous verified exporter: maximum logits difference
  7.63e-6, hard labels100% equal on both images. This is within-runtime numerical
  parity; TensorRT engines remain the application's responsibility.
- Source distribution and wheel built successfully. The wheel was installed into
  an isolated directory and imported from outside the source tree. All five CLI
  help commands and strict checkpoint loading passed; prepare checked the full
  18000/2000 image-label pairs. Predict ran a native-resolution vehicle image;
  evaluate ran two images in CPU FP32 with its limited scope recorded in output.
- All13 local CPU regression tests and Ruff checks pass.
- The124-to-8 initialization route was checked with the original parent checkpoint;
  all non-classifier tensors remain unchanged. GitHub Actions is configured
  for CPU tests but has not been executed remotely.

Historical full validation results remain in `baseline.md`. No full retraining,
new2000-image evaluation, or new mapping-quality assessment is implied here.
