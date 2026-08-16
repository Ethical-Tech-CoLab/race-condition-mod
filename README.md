# Race Condition — Evacuation Twin (fork)

> **This is a fork of [`GoogleCloudPlatform/race-condition`](https://github.com/GoogleCloudPlatform/race-condition).**
> It repurposes the upstream multi-agent *marathon* simulation into a
> location-swappable *civilian evacuation* twin, and ships a set of real-city
> case-study scenarios that run entirely in the browser.
>
> The complete upstream documentation is preserved verbatim in
> **[README.upstream.md](README.upstream.md)** — read that for the backend, the
> agents, local setup (`make init` / `make start`), and cloud deployment. This
> file only describes **what this fork adds** and **what is still in progress**.

[![CI](https://github.com/Ethical-Tech-CoLab/race-condition-mod/actions/workflows/ci.yml/badge.svg)](https://github.com/Ethical-Tech-CoLab/race-condition-mod/actions/workflows/ci.yml)
[![Deploy frontend to GitHub Pages](https://github.com/Ethical-Tech-CoLab/race-condition-mod/actions/workflows/pages.yml/badge.svg)](https://github.com/Ethical-Tech-CoLab/race-condition-mod/actions/workflows/pages.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Demos

The **marathon** is the working reference simulation and the regression anchor
for everything else. The evacuation and pedestrian scenarios are **work in
progress** — the frontend renders them, but they are not yet driven by the live
multi-agent engine (see *Work in progress* below).

| Demo | Status | Where |
| --- | --- | --- |
| **Marathon — Las Vegas Strip** | ✅ **Live** | **[ethical-tech-colab.github.io/race-condition-mod](https://ethical-tech-colab.github.io/race-condition-mod/)** |
| Mariupol evacuation twin | 🚧 WIP | `?scenario=mariupol` — renders; not engine-driven |
| Paris / Barcelona / Venice / NYC | 🚧 WIP | `?scenario=paris` \| `barcelona` \| `venice` \| `nyc` |
| DUMBO pedestrian | 🚧 WIP | backend scenario pack only — no frontend scenario yet |
| NYC Marathon route | 🚧 WIP | `?scenario=nyc` — polyline is ~88% of the official distance |

The live demo is a **static build**: no backend, no API keys, no cost. It replays
recorded runs client-side, so the marathon behaves exactly as it did when
recorded.

> This fork's demo is built and published from
> [`Ethical-Tech-CoLab/race-condition-mod`](https://github.com/Ethical-Tech-CoLab/race-condition-mod).
> The parent fork's build remains at
> [yorkerhodes3.github.io/race-condition-mod](https://yorkerhodes3.github.io/race-condition-mod/).
> **Issues and the backlog still live on
> [`yorkerhodes3/race-condition-mod`](https://github.com/yorkerhodes3/race-condition-mod/issues)** —
> issues are disabled on this fork, so tracker links below intentionally point
> there.

---

## Why this fork exists

Upstream models *thousands of autonomous agents moving along a route through a
real 3-D city*. That is structurally the same problem as **civilian
evacuation**: many people, imperfect information, shared routes, bottlenecks,
stopping points, and a clock.

This fork changes *what the simulation is about* — a marathon becomes an
evacuation — **without forking the engine's behaviour**. Vegas still renders
identically; every new scenario is opt-in via a `?scenario=<id>` URL parameter
and a data pack. The design record is
[docs/DESIGN-CHANGES-SITE-Purpose.md](docs/DESIGN-CHANGES-SITE-Purpose.md)
(an RFC — see *Work in progress* below for what is and isn't built yet).

The evacuation framing is grounded in the open, IHL-anchored research of the
[Ethical Tech CoLab](https://github.com/Ethical-Tech-CoLab/mariupol-evacuation-model).
It is a **planning and education** tool over open/synthetic data — **not** an
operational tracking or targeting tool. See *Fidelity & ethics* below.

## What this fork adds

- **A location-swappable schematic scenario framework** —
  [`web/frontend/src/app/scenarios/`](web/frontend/src/app/scenarios/). A `Site`
  registry keyed by `?scenario=<id>`; each scenario is pure data (building
  footprints, routes, POIs, damage) rendered by a GLB-less schematic renderer
  ([`schematic-site.ts`](web/frontend/src/app/viewport/scene/schematic-site.ts)).
- **A backend scenario framework** — [`agents/scenarios/`](agents/scenarios/).
  The simulation engine is now scenario-parameterised rather than
  marathon-specific: one shared physics kernel runs a marathon, a pedestrian
  district, and an evacuation. See *How the simulation works* below.
- **A "Mariupol" evacuation twin** — real ETC/OSM geometry, a humanitarian
  corridor, danger/shelter zones, and family-cohort evacuees.
- **Four real-city case-study scenarios** (Paris, Barcelona, Venice, NYC) — real
  OpenStreetMap building footprints with an illustrative 5-zone / 2-exit /
  12,000-person evacuation model each.
- **The NYC Marathon route** — an approximate polyline of the real TCS 5-borough
  course, shipped for comparison against the upstream Vegas Strip marathon.
- **A "Walk Route" eye-level camera** — glides smoothly to the corridor start
  and rides it at eye level so you can inspect the buildings along the way.
- **Operator Console presets** for every scenario
  ([`web/frontend/public/console.html`](web/frontend/public/console.html)).

### New demo artefacts

Each scenario is a self-contained pack under
[`web/frontend/public/scenarios/<id>/`](web/frontend/public/scenarios/):

| Scenario | `?scenario=` | Status | Buildings (real OSM) | Model | Notes |
| --- | --- | --- | --- | --- | --- |
| Mariupol | `mariupol` | 🚧 WIP | ETC + OSM centroids | evacuation corridor + damage | retrospective twin |
| Paris | `paris` | 🚧 WIP | 3,842 | 5 zones / 2 exits / 12k | Marais · Île de la Cité · Bastille |
| Barcelona | `barcelona` | 🚧 WIP | 4,243 | 5 zones / 2 exits / 12k | Ciutat Vella |
| Venice | `venice` | 🚧 WIP | 4,438 | 5 zones / 2 exits / 12k | exits at the real land egress |
| NYC | `nyc` | 🚧 WIP | 2,568 | 5 zones / 2 exits / 12k | + `marathon.geojson` |

**WIP means:** the schematic geometry, routes, POIs and camera work render
today, but no scenario in this table is yet driven by the live multi-agent
engine — the gateway and tick loop still run the Vegas marathon. Wiring them up
is the top item under *Work in progress*.

Each pack carries a `README.md` documenting **sources and per-layer fidelity**,
and every registered city is guarded by
[`city-scenarios.spec.ts`](web/frontend/src/app/scenarios/city-scenarios.spec.ts)
(zone counts, population = 12,000, consistent vulnerable split, real buildings,
a corridor). A shared registry
([`city-scenarios.ts`](web/frontend/src/app/scenarios/city-scenarios.ts)) is the
source of truth for each city's targets and provenance.

## How the simulation works

Understanding this is the prerequisite for changing what the simulation is
*about*. The upstream engine is not "an LLM that moves runners" — it is a
four-layer system in which **code owns the dynamics and the model owns the
narration**.

| Layer | Decides | Where | LLM? |
| --- | --- | --- | --- |
| **L0 Identity** | who this agent is — ability, frailty, starting state | [`agents/runner/initialization.py`](agents/runner/initialization.py) | No — seeded from `sha256(session_id)` |
| **L1 Policy** | which tool to call this tick | [`runner/agent.py`](agents/runner/agent.py) (LLM) or [`runner_autopilot/autopilot.py`](agents/runner_autopilot/autopilot.py) (deterministic) | Either |
| **L2 Physics** | what actually happens to the agent | [`agents/runner/kernel.py`](agents/runner/kernel.py) | No |
| **L3 World** | clock, course, hazards, aggregation | [`agents/simulator/`](agents/simulator/) | No |

**L2 is authoritative over L1.** A policy can bias the physics kernel's inputs
but can never override its constraints. That single property is what makes the
engine safe to repoint at an evacuation: the outcome layer stays deterministic,
seeded, and reviewable no matter what any model does.

Two consequences worth knowing before you change anything:

- **Everything is reproducible from the session ID.** Agent profiles derive from
  `sha256(session_id)`, and each hydration/resource station gets its own seeded
  RNG sub-stream, so the same ID always produces the same trajectory. Runs are
  therefore auditable and citable — and testable byte-for-byte.
- **The LLM's causal footprint is small.** In the shipped prompt the model is
  instructed to call `process_tick` every tick and nothing else; its only output
  that varies is `inner_thought`, a short display string. Pacing, hydration,
  fatigue, and finishing are all computed in code.

### Scenarios

A scenario is data, not a code branch. [`agents/scenarios/spec.py`](agents/scenarios/spec.py)
defines the frozen contract — physics constants, seeded profile distributions,
personas, destinations, and a termination rule — and each pack supplies values:

| Pack | Status | Terminal destinations | Distinguishing feature |
| --- | --- | --- | --- |
| [`marathon.py`](agents/scenarios/marathon.py) | ✅ reference | one finish line | the regression anchor; every value imported from `runner/constants.py` |
| [`dumbo.py`](agents/scenarios/dumbo.py) | 🚧 WIP | A/C High St → F York St → ferry | walking pace; accessibility routing; a schedule-constrained ferry |
| [`mariupol.py`](agents/scenarios/mariupol.py) | 🚧 WIP | corridor exits → shelters | hazards that close exits; household personas |

The unifying abstraction is the **destination**: a marathon water stop, a subway
entrance, a shelter, and an aid post are one type, differing by whether they end
the run, how they rank, whether they have capacity, and when they are open.
Closing a destination is how hazards and timetables remove options — agents fall
through to the next-ranked choice
([`goals.py`](agents/scenarios/goals.py)).

Marathon is deliberately the *degenerate* case — one terminal destination, no
personas — which makes it a strict correctness test: if the generalised model
reproduces marathon byte-for-byte, the abstraction is right. That is enforced by
a committed golden fixture
([`test_golden_marathon.py`](agents/tests/test_golden_marathon.py)).

### Hazards and agent attention

[`hazard.py`](agents/scenarios/hazard.py) exists because "an agent walking a
straight road has no decision to make" is only true until something happens *to*
it. Decisions are triggered by a change in the agent's **information state**, not
by road topology.

- **Perception, not omniscience.** Agents learn about a hazard through sight,
  sound, broadcast alerts, or word of mouth — each seeded, so awareness is
  deterministic but uneven. Agents respond to what they *know*, not to ground
  truth.
- **Salience decides who thinks.** A weighted score (dominated by time-to-impact)
  determines whether an agent needs to reason or can keep walking
  deterministically.
- **A budget prevents a thundering herd.** A hazard alerts everyone in radius on
  the *same* tick. `PromotionBudget` broadcasts a salience threshold so only ~N
  agents reason per tick — admission control without coordination.

### Narrative

[`narrative.py`](agents/scenarios/narrative.py) supplies agent inner monologue
from a **reviewed static corpus** indexed by the same seeded RNG, rather than by
per-agent inference. `inner_thought` is non-causal and is produced far faster
than it can be displayed, so pooling costs nothing measurable. For the evacuation
scenario this is the *correct* design rather than a cost saving: a fixed corpus
can be reviewed before anyone sees it, and no model can improvise about real
events. That corpus ships gated (`reviewed=False`) and renders nothing until
signed off.

## Testing this fork's backend changes

```bash
uv sync --frozen                                    # see note below
uv run pytest agents/ -q -m "not slow and not integration"
```

Three suites matter for scenario work:

| Suite | Guarantees |
| --- | --- |
| [`agents/tests/test_golden_marathon.py`](agents/tests/test_golden_marathon.py) | marathon output is unchanged, byte-for-byte |
| [`agents/scenarios/tests/test_all_scenarios.py`](agents/scenarios/tests/test_all_scenarios.py) | every scenario satisfies the same invariants |
| [`agents/runner/tests/test_llm_contract_guard.py`](agents/runner/tests/test_llm_contract_guard.py) | the LLM runner's prompt/state contract still holds |

Adding a scenario means adding one entry to `ALL_SCENARIOS` and inheriting the
whole invariant suite. Re-blessing the golden fixture is deliberate and should
carry justification:
`GOLDEN_BLESS=1 uv run pytest agents/tests/test_golden_marathon.py -k bless`.

> **Behind a corporate proxy?** `uv sync --frozen` resolves the lockfile's pinned
> `files.pythonhosted.org` URLs and will fail if those are blocked. Do **not**
> re-lock (it breaks CI parity). Install through your internal index instead:
> `uv pip install --index-url <internal-index> -e .` plus the `dev` group. On
> Windows also set `PYTHONIOENCODING=utf-8`, or one test that prints Unicode
> histograms fails under cp1252.

## How it works offline today


The whole point of the fork's deliverable is that it runs with **zero backend**:

1. **Static build on GitHub Pages.** The Angular frontend is built in CI
   ([`.github/workflows/pages.yml`](.github/workflows/pages.yml)) and served as a
   static site. (The build runs in CI because the local npm registry is blocked
   behind a corporate feed.)
2. **Cached replay** (inherited from upstream). The default mode replays recorded
   NDJSON runs — real timing, real agent output, no network.
3. **Schematic scenarios are pure data.** Buildings (`buildings.json`), routes
   and POIs (`*.geojson`), and damage (`damage.json`) are fetched and rendered
   client-side. No LLM, no gateway, no database — the evacuation *visualisation*
   is fully self-contained.

So today the fork delivers the **rendered scenarios and the camera/console UX**
offline. It does **not yet** drive those scenarios from the live multi-agent
backend (see below).

## Dependencies on other systems

| Capability | Depends on | Needed for |
| --- | --- | --- |
| Rendered scenarios + cached replay | Nothing (static site) | The offline demo — works today |
| Building geometry | OpenStreetMap via Overpass API (ODbL) | Regenerating scenario packs (already vendored) |
| Mariupol data | [Ethical Tech CoLab](https://github.com/Ethical-Tech-CoLab/mariupol-evacuation-model) (open data) | The Mariupol twin |
| Live multi-agent mode | Upstream backend: Go gateway, Python ADK agents, Vertex AI, AlloyDB, Redis, Pub/Sub | Driving any scenario live (upstream feature) |
| Route planning with live maps | Google Maps MCP + API key (upstream) | The planner's geographic routes |
| Static deploy | GitHub Actions + GitHub Pages | Publishing the offline demo |

Satellite imagery is **link-out only** (Esri Wayback provider terms); this fork
does not re-host tiles.

## Work in progress — what's left (mostly backend)

The **frontend** evacuation *visualisation* is implemented and shipping. The
**backend** now has a scenario framework and three working scenario packs
(see *How the simulation works*), but those packs are **not yet driving the live
simulator** — the gateway and ADK tick loop still run the Vegas marathon. Open
threads, roughly in priority order:

- **Wire scenarios into the live engine.** The scenario framework
  ([`agents/scenarios/`](agents/scenarios/)) is built and tested, but
  `prepare_simulation` / `spawn_runners` still assume a marathon. Make the
  simulator select a `ScenarioSpec`, pass it through the tick event, and let the
  kernel run it end to end.
- **Local density field.** Agents currently receive an identical broadcast and
  read only their own state — there is **no agent-to-agent interaction at all**.
  A per-segment density array in the tick event is the smallest change that makes
  agents affect each other, and it is the prerequisite for queueing at exits,
  crowd-crush, and word-of-mouth hazard propagation (the perception term already
  degrades to zero without it).
- **Destination capacity and queueing.** `capacity` is declared on every
  destination but not yet enforced. The ferry is the clearest gap — a ferry is a
  *vehicle*, not a doorway: passengers accumulate, board in a batch, and depart
  together. Tracked in `FERRY_MODELLING_GAPS` in
  [`dumbo.py`](agents/scenarios/dumbo.py).
- **Evacuation agent semantics.** Household cohorts that move together, wait for
  each other, and mingle at decision points; movement mode (foot/car/bus/train).
  Personas declare `group_size`, but cohesion is not yet simulated. Tracked in
  `MODELLING_LIMITS` in [`mariupol.py`](agents/scenarios/mariupol.py).
- **Causal LLM decisions.** Narrative pooling is done; agent *reasoning*
  (`choose_route`, `turn_back`) is not. The seam exists — any
  OpenAI-compatible endpoint via `RUNNER_MODEL=openai/...` and `OPENAI_API_BASE`
  — and `PromotionBudget` already caps how many agents may reason per tick.
  Decisions should be cached per cohort, since agents in the same
  (persona, segment, hazard, options) cell face the same problem.
- **Narrative review sign-off.** The evacuation corpus is authored but ships
  gated (`reviewed=False`) and renders nothing until reviewed by someone with the
  relevant expertise.
- **2-D / network movement.** Position is a scalar distance along one route.
  Free pedestrian movement and route *choice* need a graph, not a line. The
  frontend already isolates this behind `IMapPath.getPositionAt(t)`.
- **DTSF twin control plane.** Drive the deployed app's REST control plane
  ([`dtsf/packs/race-condition/`](dtsf/packs/race-condition/)) so Live mode can
  target an evacuation scenario without a rebuild. Its `ollama` twin is also the
  intended route to a deterministic, offline LLM path.
- **Frontend/backend scenario coupling.** The frontend hard-codes marathon
  assumptions (`distance / 42195`, a `runner_autopilot` telemetry filter), so
  backend scenarios cannot yet render.
- **Building visual fidelity** — tracked as
  [**VIS-1 (#20)**](https://github.com/yorkerhodes3/race-condition-mod/issues/20)
  and in [docs/BACKLOG.md](docs/BACKLOG.md): real OSM heights → footprint
  polygons → satellite drape → photorealistic 3D tiles. Today buildings are real
  footprints with **hashed heights only** (no facade/satellite texture).
- **Marathon distance fidelity.** The NYC Marathon polyline is ~37.4 km of
  straight chords vs. the official 42.195 km (~88%); the surveyed GPX centreline
  would close the gap.

See [docs/BACKLOG.md](docs/BACKLOG.md) and the
[issue tracker](https://github.com/yorkerhodes3/race-condition-mod/issues) for
the running list.

## Fidelity & ethics

- **Real:** OpenStreetMap building footprints (© OpenStreetMap contributors,
  ODbL). Every pack's `README.md` labels each layer HIGH / MEDIUM / illustrative.
- **Illustrative / synthetic:** zones, exits, evacuation corridors, and all
  demographics (the 12,000-person split, ~40% vulnerable). Building heights are
  hashed, not real storeys.
- **Boundaries:** a planning/education modelling exercise over open/synthetic
  data. **Not** operational; must not ingest live individual-location data; must
  not be used to locate or interdict people. Mirrors the ETC stance. See
  [docs/P7-MARIUPOL-PREP.md](docs/P7-MARIUPOL-PREP.md) §5.
- **Enforced in code, not just stated.** For the evacuation scenario these are
  assertions, not conventions:
  - **Attrition is out of scope.** `MARIUPOL_PHYSICS.collapse_threshold` is `0.0`,
    making "collapsed" unreachable, and a test drives an agent at zero resource
    for 40 ticks to prove it. Hazards affect **routing and attention only** —
    another test asserts no hazard or casualty field ever reaches the simulation
    payload.
  - **Narrative is reviewed, never improvised.** The evacuation corpus ships
    `reviewed=False` and renders nothing until signed off; a test enforces the
    gate, and another rejects graphic vocabulary.
  - **Runs are reproducible.** Everything derives from seeded RNG, so any result
    can be re-run and audited rather than taken on trust.

## Relationship to upstream

This is a derivative work under the **Apache License 2.0** (see [LICENSE](LICENSE)).
The upstream project, its architecture, and all original credit belong to the
Race Condition team at Google — see [README.upstream.md](README.upstream.md) and
its Contributors section. This fork's changes are additive and keep the upstream
Vegas marathon demo render-identical.

- **Run the full stack (backend, agents, cloud):** follow
  [README.upstream.md](README.upstream.md).
- **Just see the offline scenarios:** open the
  [live demo](https://ethical-tech-colab.github.io/race-condition-mod/) or run
  the frontend (`web/frontend`) with `npm ci && npm start`.
- **Understand the design:** [docs/DESIGN-CHANGES-SITE-Purpose.md](docs/DESIGN-CHANGES-SITE-Purpose.md),
  [docs/CONSOLE-REFERENCE.md](docs/CONSOLE-REFERENCE.md), and each pack's `README.md`.
- **Change agent behaviour:** start at *How the simulation works* above, then
  [`agents/scenarios/spec.py`](agents/scenarios/spec.py) (the contract) and
  [`agents/runner/kernel.py`](agents/runner/kernel.py) (the dynamics).
