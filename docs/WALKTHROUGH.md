# SolarSight walkthrough (file by file)

Read this once top to bottom and you can explain every file in the repo.

## The 60-second version

An inspector uploads a thermal image. YOLO (a convolutional neural network)
returns bounding boxes around faults. We convert those boxes into plain
Python records, draw them on the image, and (on click) store them in
Supabase's Postgres. The History screen reads them back into charts and CSV.

## File-by-file

### `app.py` (entrypoint)
Sets the page config, renders the sidebar, and routes to the chosen screen.
It stays tiny on purpose: a reader sees the whole app shape in 20 lines.
The model is *not* loaded here - only the Upload screen ever needs it.

### `src/config.py`
One module holding paths (`models/best.pt`, `data/`), thresholds (confidence
0.40, NMS IoU 0.45), and the 8 class names in dataset order. Central config
means the class list exists in exactly one place instead of drifting between
the pipeline, the UI and the charts.

### `src/pipelines/detection.py` (all ML logic)
* `Detection` - frozen dataclass: one fault with class, confidence, pixel
  box, optional track id.
* `results_to_detections` - converts one ultralytics `Results` object into
  `Detection`s. It duck-types the input (`.boxes.xyxy/.conf/.cls` + `.names`)
  so tests inject fakes; it also handles `boxes=None` (ultralytics' way of
  saying "nothing found") which would crash naive code.
* `run_detection` - predicts with conf/iou thresholds and filters.
* `dedupe_by_track` - video mode: keeps each track's best appearance.
* `load_model` - **the only place that imports ultralytics**. Lazy import =
  pytest/CI never load torch.

### `src/pipelines/video.py` (stretch goal)
`iter_video_detections` walks frames with OpenCV, calls
`model.track(persist=True)` so ByteTrack assigns stable ids across frames,
and yields `(frame_index, frame, detections)`. A generator so the UI can show
progress instead of freezing. `summarize_tracks` collapses frames into one
record per faulty panel.

### `src/ui/draw.py`
PIL drawing: colored boxes, a label plate (class + confidence, plus
`#track_id` in video mode), deterministic per-class colors. Pure function -
numpy in, PIL image out - so it is testable without Streamlit.

### `src/components/`
* `sidebar.py` - navigation, site name, threshold sliders, and the model
  status banner (a friendly warning if `best.pt` is missing, not a crash).
* `results_table.py` - per-class metric tiles + sortable detection table,
  with an explicit "no faults" empty state.
* `export.py` - pure CSV helpers (rows -> DataFrame -> bytes) so export is
  unit-testable without Streamlit.

### `src/screens/upload.py`
The image flow: decode bytes with PIL -> predict -> draw -> table -> optional
save. The video flow: temp file (OpenCV needs a path) -> progress-tracked
tracking loop -> dedupe by track -> results. The save block checks
`is_configured()` first: no secrets = a polite info box, never a crash.

### `src/screens/history.py`
Reads inspections (filters applied *in Postgres*), shows tables, two charts
(faults by class; faults per inspection), the CSV download button, and a
per-inspection drill-down of stored detections.

### `src/database/db.py` (all SQL lives here)
`InspectionsDB` wraps the supabase-py fluent API and takes the client via
constructor injection - tests pass the in-memory fake from conftest.py.
Row mapping (`detection_row`) is separated from insertion, and
`save_inspection` sequences parent + children inserts. `is_configured()`
lets the UI degrade gracefully without secrets.

### `supabase/schema.sql`
Two tables, three indexes, RLS with anon INSERT/SELECT-only policies (this is a no-auth demo;
a real app would key policies to `auth.uid()`). `detections.inspection_id`
is `ON DELETE CASCADE` so deleting an inspection cleans up its detections
(deleting needs the service-role key; the anon key used by the app cannot delete).

### `training/download_data.py`
Downloads the Kaggle dataset anonymously via kagglehub, copies it to
`data/ImageSet`, then **verifies** split counts, label pairing and the class
list against the documented facts, printing the per-class table. It fails
loudly if the dataset ever changes - docs/DATA.md can't silently go stale.

### `training/train_yolo.py`
Device detection printed up front (never silently train on CPU when you
expected a GPU), CLI overrides (epochs/batch/imgsz/workers), fixed seed for
reproducibility, optional `--export-model` to copy best weights into
`models/`.

### `training/evaluate.py`
Runs `model.val` on the held-out test split and writes per-class
precision/recall/mAP tables (md + csv), the confusion matrix, PR curves and
a prediction grid into `docs/assets/`. Docs quote its output; nothing typed
by hand.

### `tests/`
Fakes, not mocks of the real libraries: `FakeBoxes`/`FakeResult` mimic the
torch tensor surface (`.cpu().numpy()`), `FakeSupabaseClient` implements the
exact fluent chain db.py uses. 33 tests cover conversion, filtering, empty
cases, dedupe, row mapping, aggregation, CSV round-trip, drawing, and two
AppTest boot tests of the real app.py.

## Data-flow walkthrough (image -> database -> chart)

1. Inspector uploads `panel_042.jpg` (640x640 thermal).
2. `run_detection` -> YOLO predicts 3 candidate boxes; one is below the 0.40
   confidence slider and is dropped. 2 `Detection`s remain.
3. `draw_detections` renders them (SingleHotSpot red, SingleDiode green).
4. Inspector clicks "Save inspection to Supabase" -> `save_inspection`
   inserts:
   * `inspections`: (id=uuid, site_name="Rooftop A",
     source_filename="panel_042.jpg", media_type="image",
     model_version="yolo:best.pt", total_faults=2, created_at=now())
   * `detections`: 2 rows, each with class_name/confidence/box and NULL
     track_id (images never have one).
5. History screen: `list_inspections` pulls the row; `fault_counts` folds
   detections into `{"SingleHotSpot": 1, "SingleDiode": 1}`; the bar chart
   renders; "Download CSV report" serves `to_csv_bytes(df)`.

## Supabase setup (the only manual step)

1. Create a free project at https://supabase.com (any region).
2. SQL Editor -> New query -> paste ALL of `supabase/schema.sql` -> Run.
   You should see `inspections` and `detections` under Table Editor.
3. Project Settings -> API: copy the **Project URL** and the **anon public**
   key.
4. `cp .streamlit/secrets.toml.example .streamlit/secrets.toml`, paste both
   values, save.
5. Reload the Streamlit app - the sidebar stops warning, History goes live,
   and Upload shows the save button.
