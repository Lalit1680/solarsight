# SolarSight architecture

## Big picture

```
                 ┌────────────────────────────────────────────────┐
                 │                 Streamlit app                  │
                 │                    app.py                      │
                 │   sidebar ─► screen routing (Upload | History) │
                 └───────────┬───────────────────────┬────────────┘
                             │                       │
              ┌──────────────▼──────────┐  ┌─────────▼──────────┐
              │  screens/upload.py      │  │ screens/history.py │
              │  image + video flows    │  │ filters, charts,   │
              │  save-to-Supabase block │  │ CSV export         │
              └──────┬─────────────┬────┘  └─────────┬──────────┘
                     │             │                 │
      ┌──────────────▼──┐   ┌──────▼──────────┐      │
      │ pipelines/      │   │ components/     │      │
      │ detection.py    │   │ results_table   │      │
      │ video.py        │   │ sidebar         │      │
      │ (ultralytics,   │   │ export (CSV)    │      │
      │  lazy import)   │   │ draw (PIL)      │      │
      └─────────────────┘   └─────────────────┘      │
                                                     │
                                          ┌──────────▼──────────┐
                                          │  database/db.py     │
                                          │  ALL Supabase calls │
                                          └──────────┬──────────┘
                                                     │
                                          ┌──────────▼──────────┐
                                          │ Supabase Postgres   │
                                          │ inspections         │
                                          │ detections          │
                                          └─────────────────────┘
```

## Data flow (upload to chart)

1. **Upload**: the inspector drops a thermal image (or drone video). Bytes
   never touch disk (except video, which OpenCV needs as a temp file path).
2. **Detection**: `pipelines/detection.run_detection` feeds the image to
   YOLO11-nano (`models/best.pt`, loaded once per process via
   `st.cache_resource`). Ultralytics applies NMS (IoU threshold from the
   sidebar slider); `results_to_detections` converts normalized tensors into
   plain `Detection` records with absolute pixel boxes; the confidence filter
   is applied.
3. **Video variant**: `pipelines/video.iter_video_detections` runs
   `model.track(persist=True)` per frame; ByteTrack assigns stable
   `track_id`s; `summarize_tracks` keeps only each track's
   highest-confidence appearance so every faulty panel is counted once.
4. **Display**: `ui/draw.draw_detections` renders boxes with PIL;
   `components/results_table` shows metrics + a sortable table.
5. **Persistence** (explicit button, image metadata only):
   `save_inspection` inserts one `inspections` row and N `detections` rows
   through `database/db.py` - the only module that knows Supabase exists.
6. **History**: `db.list_inspections` (filtering happens in PostgREST),
   `db.fault_counts` aggregates per class, `components/export` turns rows
   into CSV for download.

## Module boundaries (the rules the code follows)

| Module | May import | Must never import |
|---|---|---|
| `pipelines/*` | numpy, config | streamlit, supabase, pandas |
| `ui/draw.py` | PIL, numpy, pipelines | streamlit, supabase |
| `components/*` | streamlit (except export), pipelines | ultralytics |
| `screens/*` | anything in src | ultralytics directly |
| `database/db.py` | streamlit, supabase (both lazily) | ultralytics |
| `app.py` | streamlit + screens | everything else directly |

**Ultralytics is imported in exactly one function** (`detection.load_model`).
That is why pytest and CI never need to load torch: tests inject fake result
objects into `results_to_detections` / `run_detection`.

## Design decisions (and their trade-offs)

* **Single Streamlit process, no API layer.** For a portfolio demo the
  Streamlit server *is* the backend. Trade-off: no multi-user concurrency or
  horizontal scaling; documented in roadmap.md.
* **Metadata-only persistence.** Images are analysed in RAM and discarded -
  no blob storage costs or privacy surface. Trade-off: History cannot show
  the original photo again.
* **Save is a manual button**, not automatic. The inspector stays in control
  of what enters the database. Trade-off: possible to forget; deliberate.
* **Aggregation in Python, filtering in Postgres.** Row counts are tiny, and
  real GROUP BY via PostgREST would need an RPC function. If data grows,
  `fault_counts` is the one method to swap for a view/RPC.
* **UUIDs from Postgres**, not client-generated ids - no coordination needed.
* **Fake-friendly seams.** `InspectionsDB(client)` and the duck-typed result
  conversion exist so tests exercise real app logic without torch or
  network. This is DI in miniature, not framework magic.
