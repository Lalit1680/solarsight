# SolarSight roadmap

## Not built yet (honest list)

* **No user accounts / auth.** Anyone with the URL and the anon key can read
  saved inspections. A real deployment would use Supabase Auth + RLS policies
  keyed to `auth.uid()`.
* **No image storage.** Annotated outputs are shown live but not persisted;
  History cannot re-display the original media.
* **No site-level geolocation** (GPS tags from drone footage, per-string
  mapping).
* **No temperature calibration.** Thermal pixels are treated as grayscale
  intensity only - no emissivity/ambient correction.
* **No API layer** beyond Streamlit itself (see architecture.md).
* **No model retraining loop in-app** and no data drift monitoring.
* **English-only UI**; no i18n.

## Near-term improvements

1. Swap the reduced CPU-trained weights for the Colab GPU model (see
   training/colab_train_gpu.ipynb) and re-run evaluate.py.
2. Persist a thumbnail per inspection (Supabase Storage) so History can show
   what was inspected.
3. Per-class confidence thresholds (hotspots are reliable at 0.4; rare
   classes need lower cutoffs or more data).
4. Site-level split evaluation to measure true generalization (DATA.md
   limitation #3).

## Bigger ideas

* **Active learning**: route low-confidence detections to a review queue,
  retrain periodically.
* **String-level analytics**: join detections with the plant's string layout
  to prioritize maintenance orders.
* **Model comparison dashboard**: train s/m variants and A/B them on the same
  test split (needs GPU; runs in Colab).
* **Streaming video**: RTSP ingest from a docked drone instead of file upload.
