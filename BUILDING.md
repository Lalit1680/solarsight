# Building & Running SolarSight from Scratch

Everything below was executed and verified on a CPU-only laptop (Fedora 44,
8-core i3-1215U, 7.4 GB RAM, no GPU). Nothing here is theoretical.

## 1. Requirements

- Python **3.11** (tested on 3.11.16; torch wheels do not yet support 3.13/3.14)
- ~4 GB free disk (dataset) + ~2 GB (packages)
- No GPU needed for inference; training on CPU takes ~9 min/epoch at 320 px

## 2. Create an environment

```bash
conda create -n solarsight python=3.11 -y
conda activate solarsight

# CPU torch FIRST - plain pip pulls a multi-GB CUDA build and can time out
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

(venv works too: `python3.11 -m venv .venv && source .venv/bin/activate`.)

## 3. Get the model weights

The trained detector (`models/best.pt`, 5.4 MB, YOLO11n fine-tuned) ships in
the release zip. If you only have the git repo, either:

- train it yourself: `python training/train_yolo.py` (see section 5), or
- download from the GitHub Releases page of this project.

The app checks for `models/best.pt` and shows a setup notice if it is missing.

## 4. Run the app

```bash
streamlit run app.py --server.port 8601
# health check:
curl http://localhost:8601/_stcore/health   # -> ok
```

Open http://localhost:8601 - upload a thermal image on the Upload screen.
Saving inspections is optional: without Supabase secrets the app still
detects and draws, and shows a setup notice instead of saving.

## 5. Rebuild everything from zero (dataset -> training -> evaluation)

```bash
# dataset (anonymous download, ~366 MB, verified with a per-class table)
python training/download_data.py

# training (10 epochs, imgsz 320, batch 8, seed 42 - as documented)
python training/train_yolo.py --epochs 10 --imgsz 320 --batch 8 --workers 2

# evaluation artifacts into docs/assets/
python training/evaluate.py --conf 0.40

# sample annotated predictions
python scripts/run_samples.py --count 4
```

`training/download_data.py` places the dataset at `data/ImageSet/`
(kagglehub extracts an inner `ImageSet/` - the script handles the nesting).

## 6. Tests & CI

```bash
pytest -q                      # 33 tests, ~2.5 s, no GPU/network needed
bash scripts/ci_local.sh       # same steps as .github/workflows/ci.yml
```

## 7. Optional: Supabase persistence

1. Create a free project at supabase.com
2. SQL Editor -> paste `supabase/schema.sql` -> Run
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`,
   paste your project URL + anon key
4. Restart Streamlit - Upload screen now saves; History screen reads back,
   charts, drill-down, CSV export

## 8. Optional: Docker (written, not verified - no Docker on the build machine)

```bash
docker compose up --build
```

See docs/TESTING.md for exactly what was run and what was not.
