# Backlog / Monitoring

Lightweight running list of known issues and deferred work to monitor. Newest
first. For the larger evacuation-twin design, see
[DESIGN-CHANGES-SITE-Purpose.md](DESIGN-CHANGES-SITE-Purpose.md).

---

## MON-1 — Residual end-of-race "freeze" (monitor)

**Status:** open · low severity · UX polish
**Observed:** 2026-07-29 on the live site (demo 4, 1k-runner field).
**Symptom:** a slight freeze/hold of the field at the very end of the marathon,
just before the finish/wrap-up sequence.

**Context — already mitigated (commit `881535f` / `98711bd`):**
- 1a: the replay gap-clamp in
  [agent-gateway-message-dump.ts](../web/frontend/agent-gateway-message-dump.ts)
  (`parseAgentGatewayMsgNdjsonInterFrameReplayMeta`) now collapses long gaps to
  one tick interval when **either** endpoint is sim-timed, inside the marathon
  window `[fire_start_gun … check_race_complete]`.
- 1b: the raw tail gap in
  [1k-runners.ndjson](../web/frontend/public/assets/1k-runners.ndjson) was
  trimmed from ~16.8s to 3.0s.

**Why a small freeze remains (hypotheses to confirm):**
1. **Inherent finish-line clamp.** Runners clamp at `t=1.0` and hold status
   `running` until the backend/replay confirms `finished`
   ([runner.ts](../web/frontend/src/app/runner.ts) `tick()`), so the fastest
   runners visibly sit at the line while the pack arrives.
2. **Post-window wrap-up gaps.** Frames after `check_race_complete`
   (`compile_results`, `stop_race_collector`, trailing `model_end`s) sit
   **outside** the marathon window, so the ~2–2.7s gaps between them are **not**
   clamped and replay verbatim.
3. **The 3.0s leading gap** left by 1b is still a visible beat.

**Options when picked up:**
- Also clamp (or floor) post-`check_race_complete` wrap-up gaps.
- Trim the 1b tail gap further (e.g. ~1s) and/or compress wrap-up frame spacing.
- Start the post-finish camera/overview sequence as soon as the **leaders**
  finish, rather than waiting for the whole field
  ([viewport-lookdev.component.ts](../web/frontend/src/app/viewport/viewport-lookdev.component.ts)
  `startPostFinishSequence` / `_simAllFinished`).
- Make runners visually **cross and exit** the finish line instead of piling on
  it at `t=1.0`.

**How to reproduce / measure:** load the site (default demo 4), watch the last
~20s; or inspect tail inter-frame gaps in the recording (search for the
`process_tick → model_start → check_race_complete` transition).

---

## VIS-1 — Building visual fidelity vs. a Google Earth view (data-limited)

**Status:** open · enhancement · data-limited (needs a licensed source)
**Tracking:** [Ethical-Tech-CoLab/race-condition-mod#2](https://github.com/Ethical-Tech-CoLab/race-condition-mod/issues/2)
(migrated from [yorkerhodes3/race-condition-mod#20](https://github.com/yorkerhodes3/race-condition-mod/issues/20))
**Raised:** 2026-07-31, after the city case-study scenarios landed
(Paris `82b848f`, Barcelona `d3a347b`, Venice `5cf4b42`, NYC `c7ce359`).

**One honest open item (not fabricated).** The deeper "inspect visuals vs. a
Google Earth view / extrapolate satellite or Street View onto the buildings"
request is only partially met. The schematic scenes currently render **real
building _footprints_ with hashed heights only** — there is **no facade or
satellite texture**, and satellite imagery is **link-out only** (Esri Wayback
provider terms; we link rather than re-host). Closing that gap would need a
licensed **3D-tiles / photogrammetry source** (e.g. Google Photorealistic 3D
Tiles, Cesium ion, or equivalent), which is **not vendored here**. Flagged as
data-limited rather than claiming a fidelity we don't have.

**What is real today (per city `public/scenarios/<id>/README.md`):**
- Buildings: **HIGH** — real OpenStreetMap footprint centroids (Overpass, ODbL).
- Building heights: **synthetic** — hashed skyline variation, not real storeys.
- Zones / exits / corridors / demographics: **illustrative** (documented).
- NYC Marathon (`nyc/marathon.geojson`): **MEDIUM** — approximate 45-waypoint
  polyline; ~37.4 km straight-chord vs. the official 42.195 km (~88%), shape and
  borough sequence faithful, absolute distance approximate.

**Options when picked up (in rough order of effort/fidelity):**
1. **Real heights, no texture (cheap, high value):** pull OSM
   `building:levels` / `height` tags in the Overpass query and drive extrusion
   from them instead of the hash. Keeps ODbL, no new provider.
2. **Footprint polygons instead of centroids:** fetch full `way` geometry
   (`out geom;`) so buildings extrude their true outline, not a box at a point.
3. **Satellite drape (terms permitting):** evaluate a basemap/tiles provider
   whose licence allows re-hosting/overlay; drape orthoimagery on the ground
   plane and/or building tops.
4. **Photorealistic 3D tiles:** integrate Google Photorealistic 3D Tiles or
   Cesium ion for true facades — highest fidelity, requires API key, cost, and
   licence review; would replace the schematic renderer for these scenarios.
5. **Marathon distance:** trace the surveyed GPX centreline to close the ~12%
   length gap if exact distance matters.

**Guardrail:** whatever lands must keep the **Vegas scenario render-identical**
(all city behaviour stays gated on `schematicFocus` / the GLB-less path) and
must not overstate fidelity — keep the per-layer honesty in each README and in
`scenarios/city-scenarios.ts` `fidelity`.

**Where to look:**
- Renderer: [schematic-site.ts](../web/frontend/src/app/viewport/scene/schematic-site.ts)
  (`addBuildings` height hashing, footprint scale).
- Data + parsing: [geo.ts](../web/frontend/src/app/scenarios/geo.ts)
  (`parseBuildings` accepts `[lon,lat(,h)]`; a height channel already exists).
- Provenance/fidelity: [city-scenarios.ts](../web/frontend/src/app/scenarios/city-scenarios.ts)
  and each `public/scenarios/<id>/README.md`.
