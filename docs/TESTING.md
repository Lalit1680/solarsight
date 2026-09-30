# Testing: what was run, what it produced, what is NOT covered

Rule for this page: every command below was executed on this machine
(Fedora 44, Python 3.11.16, 8-core i3-1215U, no GPU) on 2026-09-29, and the
outputs are pasted from the real runs. Where something could not be run in
this environment, it says so explicitly.

## Commands and real outputs

### 1. Unit + smoke tests

```
$ python -m pytest -v
============================= test session starts ==============================
platform linux -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
rootdir: /path/to/SolarSight
configfile: pytest.ini
testpaths: tests
collected 33 items

tests/test_app_smoke.py ..                                               [  6%]
tests/test_database.py ............                                      [ 42%]
tests/test_detection_pipeline.py ........12                              [ 78%]
tests/test_export_and_draw.py .......                                    [100%]

============================== 33 passed in 2.51s ==============================
```

(33 passed, ~2.5 s. The AppTest tests boot the REAL app.py headlessly - no
secrets, no model weights required.)

### 2. Dataset verification (reproducible check of docs/DATA.md)

```
$ python training/download_data.py
Verification passed: splits, label files and class list all match.

Per-class bounding-box instances (from label files):
class                      train   valid    test
MultiByPassed                357      34       0
MultiDiode                   969      77       1
MultiHotSpot                5081     209       7
SingleByPassed              2978     263      11
SingleDiode                 1844     128      10
SingleHotSpot               7311     323       2
StringOpenCircuit           1197      69       0
StringReversedPolarity       108      16       0
```

### 3. Local CI (same steps as .github/workflows/ci.yml)

```
$ bash scripts/ci_local.sh
== 1. Dependency consistency ==
No broken requirements found.
== 2. Tests ==
33 passed
== 3. Import check without secrets ==
app module imports cleanly without secrets
ALL CI STEPS PASSED
```

### 4. App actually serves (health endpoint)

```
$ streamlit run app.py --server.port 8601 --server.headless true &
$ curl -i http://localhost:8601/_stcore/health
HTTP/1.1 200 OK
ok
```

### 5. Real model on held-out test images

```
$ python scripts/run_samples.py --count 4
Model: best.pt | conf>=0.4 | 4 test images
0raw_data_total_dataset_DJI_20230809115646_0006_T_jpg.rf.c0b14af525fab877a150a44ee6bfc1f3.jpg: 3 fault(s)
   SingleHotSpot          conf=0.73 box=(500,238,539,333)
   MultiHotSpot           conf=0.66 box=(500,337,539,436)
   SingleHotSpot          conf=0.43 box=(500,337,539,435)
101MEDIA_DJI_0333_T_JPG_jpg.rf.13d79587843a1f42dac0797208e8eee8.jpg: 1 fault(s)
   MultiHotSpot           conf=0.68 box=(126,551,230,629)
101MEDIA_DJI_0343_T_JPG_jpg.rf.ea0ecc6614d149112f7a62c294c6d300.jpg: 2 fault(s)
   SingleByPassed         conf=0.53 box=(11,135,56,398)
   SingleByPassed         conf=0.49 box=(580,133,636,402)
101MEDIA_DJI_0361_T_JPG_jpg.rf.c8e41676aa3376d7f77ff631b9216600.jpg: 3 fault(s)
   MultiHotSpot           conf=0.59 box=(8,138,54,261)
   SingleHotSpot          conf=0.57 box=(254,13,299,137)
   MultiHotSpot           conf=0.40 box=(254,13,299,140)

Annotated images: docs/assets/samples/*_pred.jpg
```

### 6. Evaluation on the test split (real metrics)

```
$ python training/evaluate.py --conf 0.40
                   Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all         15         31      0.692      0.762      0.641      0.541
            MultiDiode          1          1          1          1      0.995      0.895
          MultiHotSpot          5          7      0.583          1       0.82       0.69
        SingleByPassed          6         11      0.909      0.909      0.848      0.685
           SingleDiode          5         10        0.8        0.4      0.375        0.3
         SingleHotSpot          1          2      0.167        0.5      0.165      0.132
Speed: 0.9ms preprocess, 27.3ms inference, 1.1ms postprocess per image

Artifacts: docs/assets/evaluation_metrics.{md,csv}, confusion_matrix.png,
predictions_grid.png  (full table + commentary: docs/TRAINING.md)
```

### 7. Training run (the weights shipped in models/best.pt)

```
$ python training/train_yolo.py --epochs 10 --batch 8 --imgsz 320 \
      --workers 2 --device cpu --model models/yolo11n.pt --export-model
# 10/10 epochs completed, ~94 min total on CPU
# final valid metrics (results.csv): mAP50 0.554, mAP50-95 0.368
# Best weights: runs/.../weights/best.pt -> copied to models/best.pt
```

## Coverage: what the tests actually verify

* YOLO result -> Detection conversion (values, dtypes, track ids)
* `boxes=None` empty-detection edge case (crashes naive code)
* confidence filtering incl. exactly-at-threshold inclusivity
* track dedup (best-confidence per track; frame-only rows ignored)
* DB row mapping (JSON-serializable), inserts, empty-insert no-op,
  filtered reads, per-class aggregation, delete
* CSV export round-trip (DataFrame -> bytes -> pandas read-back)
* drawing (output size, original image untouched, track labels, palette wrap)
* two AppTest boots of the real app (Upload and History screens)

## What is NOT covered (honest list)

* **Real model quality** - that is evaluate.py's job on the real test split,
  not unit tests with fakes.
* **ByteTrack across real frames** - tracking needs the real tracker; the
  dedup logic is tested, the tracker itself is trusted.
* **Network I/O to Supabase** - fakes do not validate RLS policies, network
  errors, or the REST layer.
* **Docker runtime** - see below.

## Failures found while testing (and fixed)

1. `training/download_data.py` - `ModuleNotFoundError: No module named 'src'`
   when run as a file. Fix: insert project root into `sys.path` before
   importing `src`.
2. Same script - archive copied as `data/ImageSet/ImageSet/...` (double
   nesting) because the kagglehub cache contains an `ImageSet/` folder.
   Fix: copy the inner `ImageSet` directory; verification then failed loudly
   until the copy was correct (that is the verification doing its job).
3. `tests/test_app_smoke.py` - `AppTest.from_file("app.py")` resolves
   relative paths against the test file, not the cwd. Fix: absolute path via
   `Path(__file__).parent.parent`.
4. `tests/conftest.py` - fake Supabase client ignored `.in_()` filters, so
   `fault_counts(inspection_ids=[...])` returned unfiltered counts. Fix:
   implement `in` filtering in the fake.
5. Two tests referenced `Detection` without importing it (`NameError`).
6. `src/screens/upload.py` imported `count_by_class` without using it
   (dead import, found while re-reading after test runs).
7. `training/evaluate.py` - `IndexError: index 5 is out of bounds` when
   building the per-class table: ultralytics orders per-class metric arrays
   by position among classes PRESENT in the split, not by global class id.
   Fix: map class id -> position via `ap_class_index` (3 of 8 classes are
   absent from the 15-image test split).
8. `requirements.txt` - `pip install -r requirements.txt` failed with
   `No matching distribution found for opencv-python-headless==5.0.0`
   on a fresh environment: the pin named a version that does not exist on
   PyPI (the real release is `5.0.0.93`; upstream numbers its releases with
   a third component). The bad pin had gone unnoticed because the original
   env already had opencv installed, so pip skipped the line - only a
   from-scratch install exercised it (exactly what BUILDING.md asks of
   readers). Since pip is all-or-nothing, the failed resolution aborted the
   whole install and `streamlit` was left missing. Fix: correct the pin to
   `5.0.0.93`, reinstall, re-verify (33/33 tests, model loads, Streamlit
   1.64.0 up).

## Docker status in this environment

**Docker is not installed on this machine** (`which docker` -> not found),
so the image build and health check could NOT be executed here. The
Dockerfile and docker-compose.yml are provided and follow the
python:3.11-slim + HEALTHCHECK pattern (health endpoint proven live in step
4 above - the container hits the same endpoint). To verify on any
Docker-capable machine:

```bash
docker compose up --build   # then: curl http://localhost:8501/_stcore/health
```

## CI status

`.github/workflows/ci.yml` mirrors the local CI script (deps -> pytest ->
import check). It runs on push/PR once the repo is pushed to GitHub; the
same steps pass locally (section 3).
