# Dataset: Thermal Solar PV Anomaly Detection Dataset (ThermoSolar-PV)

**Every number on this page was verified by reading the downloaded files on
2026-09-29** (via `python training/download_data.py`, which re-verifies on
every run). Nothing is quoted from memory.

| Field | Value |
|---|---|
| Name | Thermal Solar PV Anomaly Detection Dataset ("ThermoSolar-PV") |
| Source | Kaggle: https://www.kaggle.com/datasets/pkdarabi/solarpanel |
| Provenance | Roboflow project `solarpanels-odahn/anomalies-detection-blv72` v14 (stated in the bundled `data.yaml`) |
| Licence | **CC BY-NC 4.0** (Attribution-NonCommercial) - read from the Kaggle dataset page |
| Downloaded | 2026-09-29, anonymously via `kagglehub` (no Kaggle account or API key needed) |
| Images | **7,339** thermal (infrared) JPEGs, all 640x640, RGB-stored grayscale content |
| Labels | YOLO detection format: one `.txt` per image, `class x_center y_center width height` (normalized); 7 empty label files in train (no-fault images) |

## Splits (counted from the actual files)

| Split | Images | Label files | Bounding boxes |
|---|---|---|---|
| train | 6,924 | 6,924 | 19,845 |
| valid | 400 | 400 | 1,119 |
| test | 15 | 15 | 31 |

## Classes and instances per split (counted from label files)

| class id | class | train | valid | test |
|---|---|---|---|---|
| 0 | MultiByPassed | 357 | 34 | 0 |
| 1 | MultiDiode | 969 | 77 | 1 |
| 2 | MultiHotSpot | 5,081 | 209 | 7 |
| 3 | SingleByPassed | 2,978 | 263 | 11 |
| 4 | SingleDiode | 1,844 | 128 | 10 |
| 5 | SingleHotSpot | 7,311 | 323 | 2 |
| 6 | StringOpenCircuit | 1,197 | 69 | 0 |
| 7 | StringReversedPolarity | 108 | 16 | 0 |
| | **total** | **19,845** | **1,119** | **31** |

## Processing already applied by the dataset authors (from its README.txt)

* Auto-orientation (EXIF stripped), resize to 640x640 (stretch), grayscale.
* **3 augmented copies of each source image** (horizontal/vertical flip,
  90-degree rotations, ±15-degree shear, ±25% brightness). This matters:
  near-duplicates of the same panel appear in both train and valid, so
  validation metrics are optimistic relative to fully-unseen sites.

## Known limitations (be honest about these in interviews)

1. **Severe class imbalance**: StringReversedPolarity has 124 instances total
   vs 7,636 for SingleHotSpot (61x). Expect weak metrics on rare classes.
2. **Tiny official test split** (15 images / 31 boxes, and 3 of 8 classes are
   absent from it). Per-class test numbers are noisy; the valid split (400
   images) gives the more stable picture. Evaluation notes both.
3. **Augmented near-duplicates across splits** (see above) inflate metrics
   versus a site-level split.
4. **Private Roboflow provenance + NC licence**: non-commercial use only;
   keep the attribution.
5. Grayscale thermal imagery: models pretrained on COCO colour photos
   transfer imperfectly; fine-tuning is required (we do exactly that).

## Reproduce

```bash
python training/download_data.py   # downloads (kagglehub, anonymous) + verifies
```

The script fails loudly if the split counts, label pairing or class list no
longer match the table above, so this page cannot silently go stale.
