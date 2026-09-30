# Training & evaluation (all numbers from real runs)

Nothing on this page is invented. Metrics are produced by
`training/evaluate.py`; the training log excerpt is pasted from the actual
run. Sections marked **PENDING** are filled in immediately after the run
finishes - if you ever see PENDING here, the run has not completed yet.

## Hardware & environment (honest)

| | |
|---|---|
| Machine | Fedora 44 laptop, Intel i3-1215U (8 cores), 7.4 GB RAM |
| GPU | **None** (checked: `nvidia-smi` not found, `torch.cuda.is_available()` False) |
| Python / torch | 3.11.16 / 2.14.0+cpu |
| Ultralytics | 8.4.165 |

## Two-tier training strategy

Because there is no GPU here:

1. **Local CPU run (shipped weights).** YOLO11-nano, reduced settings chosen
   from a measured CPU benchmark (76 ms/image inference for yolo11n @640):
   `--epochs 10 --imgsz 320 --batch 8 --workers 2 --device cpu`.
   Purpose: a real, complete, reproducible train->evaluate->app pipeline on
   this machine, finishing in ~90 minutes.
2. **Colab GPU notebook (recommended final model).**
   `training/colab_train_gpu.ipynb` trains the same nano model at full
   640px / 60 epochs / batch 32 on a free T4 (~25 min). Steps are exact:
   Runtime > T4 GPU > Run all > download best.pt into models/.

## Dataset

See [DATA.md](DATA.md). In short: 7,339 thermal images (6,924/400/15),
8 classes, ~21k boxes, CC BY-NC 4.0. Downloaded and verified with
`training/download_data.py`.

## Hyperparameters (local run)

| Parameter | Value | Why |
|---|---|---|
| base model | yolo11n.pt (COCO) | transfer learning; nano trains fast on CPU |
| epochs | 10 | CPU budget; loss still falling at end (documented) |
| imgsz | 320 | 2x2 fewer pixels than 640 -> ~4x faster epochs; costs small-defect recall |
| batch | 8 | fits RAM comfortably with workers=2 |
| lr0 | 0.01 (ultralytics auto, SGD) | standard fine-tune start |
| fliplr / flipud | 0.5 / 0.0 | panels have a natural "up" in drone footage |
| mosaic | 1.0 (default) | helps small hotspots |
| patience | 15 | early stopping if no improvement |
| seed | 42 | reproducible shuffling |
| val split | valid (400 imgs) | per-epoch monitoring |

## Training log (real, pasted)

```
Ultralytics 8.4.165 🚀 Python-3.11.16 torch-2.14.0+cpu CPU (12th Gen Intel Core i3-1215U)
engine/trainer: ... epochs=10, imgsz=320, batch=8, device=cpu, workers=2, seed=42 ...
```

Full 10-epoch curve (runs/solar/.../results.csv, copied to
[assets/training_results.csv](assets/training_results.csv)):

| epoch | train box_loss | valid mAP50 | valid mAP50-95 |
|---|---|---|---|
| 1 | 1.864 | 0.127 | 0.082 |
| 2 | 1.693 | 0.195 | 0.115 |
| 3 | 1.621 | 0.267 | 0.167 |
| 4 | 1.573 | 0.327 | 0.197 |
| 5 | 1.521 | 0.360 | 0.226 |
| 6 | 1.480 | 0.442 | 0.289 |
| 7 | 1.464 | 0.453 | 0.294 |
| 8 | 1.427 | 0.515 | 0.336 |
| 9 | 1.392 | 0.516 | 0.342 |
| 10 | 1.374 | **0.554** | **0.368** |

Losses fall steadily and valid mAP50 was still improving at epoch 10
(0.516 -> 0.554 over the last two epochs) - the run is undertrained by
design (CPU budget), which is exactly why the Colab 60-epoch notebook
exists. Total training time: ~94 minutes (5631 s).

## Test-split evaluation

Command (executed after training completed; the same command reproduces it):

```
python training/evaluate.py --conf 0.40
```

Artifacts written to docs/assets/:
`evaluation_metrics.md`, `evaluation_metrics.csv`, `confusion_matrix.png`,
`predictions_grid.png`.

**Results (real output of the command above, run 2026-09-29):**

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| MultiByPassed | - | - | - | - |
| MultiDiode | 1.000 | 1.000 | 0.995 | 0.895 |
| MultiHotSpot | 0.583 | 1.000 | 0.820 | 0.690 |
| SingleByPassed | 0.909 | 0.909 | 0.848 | 0.685 |
| SingleDiode | 0.800 | 0.400 | 0.375 | 0.300 |
| SingleHotSpot | 0.167 | 0.500 | 0.165 | 0.132 |
| StringOpenCircuit | - | - | - | - |
| StringReversedPolarity | - | - | - | - |

**Overall: precision 0.692 - recall 0.762 - mAP50 0.641 - mAP50-95 0.541**
(15 test images, 31 instances; `-` = class absent from the test ground
truth, so its metrics are undefined, not zero.)

CPU speed (from the ultralytics val output): 0.9 ms preprocess,
27.3 ms inference, 1.1 ms postprocess per image - comfortably interactive
for the app (measured 76 ms/image end-to-end with the standalone predict
path earlier in development).

### How to read these numbers honestly

* The 10-epoch/320px CPU model reaches a useful but **not production**
  mAP50 of 0.641. The colab 640px/60-epoch run is the obvious next step.
* SingleByPassed and MultiHotSpot (the well-represented, visually distinct
  faults) work well. SingleDiode recall (0.400) and SingleHotSpot AP
  (0.165) are weak - and SingleHotSpot is ironic given it dominates train
  (7.6k boxes): the TEST split contains only 2 instances of it, so these
  numbers are extremely noisy (1 bad box swings AP by ~0.5).
* This is exactly the "tiny test split" problem documented in DATA.md:
  per-class test metrics are indicative only; the 400-image valid split is
  the stable reference (final-epoch valid mAP50 = 0.554, see log above).

## Weak classes (real observations from the run above)

* **SingleHotSpot (AP50 0.165)** - despite being the most common class in
  train, only 2 test instances exist and both are hard; small hotspots also
  suffer most from the 320px reduction. Fix: full-res GPU run.
* **SingleDiode (recall 0.400)** - missed 3 of 5 images containing it at
  conf 0.40; diode patterns are subtler than hotspots/bypasses.
* **MultiByPassed, StringOpenCircuit, StringReversedPolarity** - absent from
  the test split entirely; they ARE trained (they appear in train/valid and
  the confusion matrix of the valid split), but cannot be scored here.

## What I would try next (with more compute)

1. The Colab 640px/60-epoch run - likely the single biggest win.
2. Site-level split to measure true generalization (DATA.md limitation #3).
3. Class-weighted loss or oversampling for the 3 rare classes.
4. Larger backbone (yolo11s) at 640px on GPU; compare per-class AP.
5. Test-time augmentation and per-class confidence thresholds (rare classes
   get lower cutoffs).
6. Export to ONNX + INT8 for CPU-friendly app inference.
