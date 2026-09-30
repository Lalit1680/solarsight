# SolarSight ☀️🔬

Upload a thermal (infrared) image of a solar panel and get instant fault
detection: annotated boxes, a results table, saved inspections in Supabase,
and a History screen with fault charts and CSV export. Drone video works too
- each faulty panel is counted once via tracking.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64-red)
![YOLO11](https://img.shields.io/badge/YOLO11-nano-green)
![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)

## What it looks like

*(screenshots in docs/assets/ - see "Honest section" below for status)*

## How it works

1. **Detection**: YOLO11-nano (Ultralytics), fine-tuned on the
   [Thermal Solar PV Anomaly Detection Dataset](https://www.kaggle.com/datasets/pkdarabi/solarpanel)
   (7,339 thermal images, 8 fault classes, CC BY-NC 4.0).
2. **App**: single-page Streamlit. Upload -> boxes + table -> optional save.
3. **Storage**: Supabase Postgres, metadata only (no images) - two tables,
   `inspections` and `detections` (schema in `supabase/schema.sql`).
4. **History**: filters (site, media type), bar charts, CSV export.
5. **Video (stretch)**: ByteTrack tracking per frame; deduplicated by
   `track_id` so each faulty panel is counted once.

## Quickstart

```bash
# 1. Environment (Python 3.11)
conda create -n solarsight python=3.11 -y && conda activate solarsight

# 2. CPU-first torch (avoids the 2 GB CUDA build on non-GPU machines)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Dataset (366 MB, anonymous download, verifies itself)
python training/download_data.py

# 4. Train (reduced CPU run) OR use the Colab notebook for a GPU run
python training/train_yolo.py --epochs 10 --imgsz 320 --batch 8 --device cpu --export-model
#    GPU: open training/colab_train_gpu.ipynb in Google Colab (free T4)

# 5. Supabase: run supabase/schema.sql in the SQL editor, then
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # paste your keys

# 6. Run
streamlit run app.py
```

Docker: `docker compose up --build` (mounts `models/` and secrets at runtime).

## Real metrics

Produced by `training/evaluate.py` on the dataset's held-out test split
(15 images / 31 boxes; 3 of 8 classes are absent from it - see
[docs/TRAINING.md](docs/TRAINING.md) for the full table, training curve and
caveats).

| model | source | mAP50 | mAP50-95 |
|---|---|---|---|
| yolo11n fine-tune | local CPU run (10 ep @ 320px, ~94 min) | **0.641** | **0.541** |
| yolo11n @ 640px/60ep | Colab GPU notebook (recommended) | not run yet | not run yet |

Per-class (test split): MultiDiode 0.995 · MultiHotSpot 0.820 ·
SingleByPassed 0.848 · SingleDiode 0.375 · SingleHotSpot 0.165 mAP50.

## Screenshots (real output, generated on this machine)

* [docs/assets/predictions_grid.png](docs/assets/predictions_grid.png) -
  held-out test images with the model's predictions
* [docs/assets/confusion_matrix.png](docs/assets/confusion_matrix.png) -
  confusion matrix from the evaluation run
* [docs/assets/samples/](docs/assets/samples) - annotated single-image runs

App screenshots (upload -> boxes -> table): run the app and they look like
the grid above - a pre-rendered UI capture was not generated in this
session.

## Tests

```bash
pytest -v          # 33 tests, no GPU/torch/network needed (fakes, not mocks)
```

CI (`.github/workflows/ci.yml`) runs the same suite on every push, plus an
import check that proves the app boots with no secrets present.

## How this was built (honest section)

This project was built with **AI-assisted development** (Codebuff, a coding
agent). The workflow was: I directed the architecture and reviewed every
file; the AI generated the code, the docs and the test suite; all results in
this repo come from commands that were actually executed on this machine
(training logs, `pytest` output, `evaluate.py` metrics, Streamlit health
checks). Dataset facts were verified by reading the downloaded files, never
quoted from memory. Where something could NOT be done in this environment,
the docs say so explicitly:

* **No GPU here** - shipped weights come from a real but reduced CPU run;
  the full-quality run is a ready-to-use Colab notebook.
* **No Docker daemon here** - image build provided but not pre-verified.
* **Supabase keys are yours to paste** - the app degrades gracefully
  (save/history disabled with a hint) until secrets exist; schema, policies
  and setup steps are in `supabase/schema.sql` and docs/WALKTHROUGH.md.
* **Not run yet**: a real save->history round-trip against a live Supabase
  project (needs your keys), and the 640px GPU training run.

**Not built yet**: auth/user accounts, image persistence, per-site
generalization metrics, temperature calibration. Full list:
[docs/roadmap.md](docs/roadmap.md).

## Licence

Code: [MIT](LICENSE). The training dataset is licensed **CC BY-NC 4.0**
(non-commercial, attribution required - see [docs/DATA.md](docs/DATA.md)), so
model weights trained on it should be treated as non-commercial too.

## Docs

* [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) - file-by-file tour + data flow
* [docs/DATA.md](docs/DATA.md) - dataset source, licence, verified counts
* [docs/TRAINING.md](docs/TRAINING.md) - hyperparameters + real metrics
* [docs/TESTING.md](docs/TESTING.md) - exact commands and real outputs
* [docs/INTERVIEW_PREP.md](docs/INTERVIEW_PREP.md) - 25 Q&A + limitations
* [docs/architecture.md](docs/architecture.md) - module boundaries, design
* [docs/roadmap.md](docs/roadmap.md) - what's next
