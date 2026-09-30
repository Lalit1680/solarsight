-- ============================================================
-- SolarSight schema
-- Paste this into: Supabase Dashboard -> SQL Editor -> New query -> Run
--
-- Design notes:
-- * We store METADATA ONLY (no images) so the DB stays small and we
--   avoid storage/privacy issues. The uploaded image is analysed in
--   memory and then discarded.
-- * detections.inspection_id -> inspections.id is ON DELETE CASCADE:
--   deleting an inspection removes its detections automatically.
-- * Row Level Security (RLS) is enabled. The anon key may only INSERT and
--   SELECT (this demo has no user accounts). It cannot UPDATE or DELETE:
--   the app never deletes, so that permission is simply not granted.
--   In a real product you would tie policies to authenticated users.
-- * The script is safe to re-run: policies are dropped and recreated,
--   tables and indexes use IF NOT EXISTS.
-- ============================================================

create table if not exists public.inspections (
    id              uuid primary key default gen_random_uuid(),
    site_name       text not null,
    source_filename text not null,
    media_type      text not null default 'image' check (media_type in ('image', 'video')),
    model_version   text not null,
    created_at      timestamptz not null default now(),
    total_faults    integer not null default 0
);

create table if not exists public.detections (
    id            uuid primary key default gen_random_uuid(),
    inspection_id uuid not null references public.inspections (id) on delete cascade,
    class_name    text not null,
    confidence    double precision not null check (confidence >= 0 and confidence <= 1),
    x1            double precision not null,
    y1            double precision not null,
    x2            double precision not null,
    y2            double precision not null,
    track_id      integer  -- only set for video; null for still images
);

-- Fast "latest inspections first" listing for the History screen.
create index if not exists idx_inspections_created_at
    on public.inspections (created_at desc);

-- Fast per-inspection detection lookups (detail view / drill-down).
create index if not exists idx_detections_inspection_id
    on public.detections (inspection_id);

-- Fast group-by-class aggregation for the History charts.
create index if not exists idx_detections_class_name
    on public.detections (class_name);

-- ------------------------------------------------------------
-- Row Level Security.
-- The app talks to Supabase with the anon key, so without RLS every
-- table would be wide open to anyone holding the key AND visible even
-- to people without it via the auto-generated REST API.
-- ------------------------------------------------------------
alter table public.inspections enable row level security;
alter table public.detections  enable row level security;

drop policy if exists "allow anon insert inspections" on public.inspections;
create policy "allow anon insert inspections"
    on public.inspections for insert to anon with check (true);

drop policy if exists "allow anon select inspections" on public.inspections;
create policy "allow anon select inspections"
    on public.inspections for select to anon using (true);

drop policy if exists "allow anon insert detections" on public.detections;
create policy "allow anon insert detections"
    on public.detections for insert to anon with check (true);

drop policy if exists "allow anon select detections" on public.detections;
create policy "allow anon select detections"
    on public.detections for select to anon using (true);
