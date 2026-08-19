# The Runner Agent Decision Model

How a simulated agent actually decides anything — and which parts of that
decision are made by a language model, which by code, and why the split falls
where it does.

This is the design record behind the scenario framework in
[`agents/scenarios/`](../../agents/scenarios/). Read it before changing agent
behaviour or adding a scenario.

---

## 1. There is no "runner schema"

A runner is not a declarative entity definition. It is the composition of four
layers that live in different files and have very different mutability:

| Layer | Decides | Implementation | LLM? |
|:--|:--|:--|:--|
| **L0 — Identity** | who this agent *is* (ability, frailty, corral) | [`runner/initialization.py`](../../agents/runner/initialization.py) | No — seeded from `sha256(session_id)` |
| **L1 — Policy** | which tool to call this tick | [`runner/agent.py`](../../agents/runner/agent.py) (LLM) or [`runner_autopilot/autopilot.py`](../../agents/runner_autopilot/autopilot.py) (deterministic) | Either |
| **L2 — Physics** | what actually happens to the body | [`runner/kernel.py`](../../agents/runner/kernel.py) | No |
| **L3 — World** | clock, course, closures, aggregation | [`agents/simulator/`](../../agents/simulator/) | No |

**The load-bearing property: L2 is authoritative over L1.** A policy can bias the
physics kernel's inputs but can never override its constraints. That single
property is what makes the engine safe to repoint at an evacuation — the outcome
layer stays deterministic, seeded, and reviewable no matter what any model does.

## 2. The three schemas that do exist

### 2a. Profile — immutable per session, generated on the first tick

`initialize_runner(state, session_id, runner_count)` draws one master ability
variable and derives everything else from it:

| Key | Distribution | Meaning |
|:--|:--|:--|
| `target_finish_minutes` | log-normal(μ=5.45, σ=0.32), clamped [120, 300] | master ability variable |
| `velocity` | `MARATHON_MI / (target/60) / SPEED_SCALE` | normalised speed; 1.0 == 6.2137 mph |
| `will_hit_wall` | Bernoulli(0.40) | whether this runner degrades sharply |
| `wall_mi` | Gauss(18.64, 1.86) | where the wall hits |
| `wall_severity` | Beta(2, 5) | multiplicative pace penalty past the wall |
| `hydration_efficiency` | bell curve on `ability_ratio = target/240` | fast and very slow deplete less; mid-pack worst |
| `crowd_responsiveness` | 75% → 0.0, else Beta(2,5) | gates the `accelerate` tool's effect |
| `wave_number`, `start_delay_minutes` | percentile → corral | staggered start |
| `water` | `100 − 8·(ability_ratio−0.5)` + N(0,1.5), clamped [88,100] | starting hydration |

Seeded by `runner_seed(session_id, salt)` = `int(sha256(f"{sid}:{salt}")[:8], 16)`
([`runner/constants.py`](../../agents/runner/constants.py)). **The same session ID
produces a byte-identical runner, forever.** The salt allows per-event
sub-streams — each resource station gets its own RNG, so stop/skip decisions are
reproducible independent of call order.

### 2b. Dynamic state — a plain dict

`velocity`, `distance`, `water`, `exhausted`, `collapsed`, `finished`,
`runner_status` ∈ {running, exhausted, collapsed, finished}, `finish_time_minutes`,
`pace_min_per_mi`, plus `_tick_params` (a staging area for the incoming event).

There is **no schema class, no pydantic model, no proto message** for a runner.
Adding fields costs nothing and requires no migration — which is what made the
scenario generalisation tractable.

### 2c. Wire protocol

[`agents/utils/runner_protocol.py`](../../agents/utils/runner_protocol.py) defines
the inbound vocabulary: `start_gun`, `tick`, `crowd_boost`, `distance_update`,
`hydration_station`, `unknown`.

**Only `tick` and `start_gun` are emitted by the simulator.** The other three are
interactive/legacy, fired from the frontend or tests. `distance_update` is now an
explicit no-op: it once caused ~4× double-depletion and universal collapse,
because `process_tick` already depletes with its own calibrated physics.

Outbound: the `process_tick` return dict, `RPUSH`ed directly to
`collector:buffer:{simulator_session_id}` in Redis, bypassing Pub/Sub.

## 3. The decision loop

```
simulator advance_tick
  └─ build_tick_event(tick, max_ticks, minutes_per_tick, elapsed_minutes,
                      race_distance_mi, collector_buffer_key, runner_count)
  └─ publish_to_runners(...)  →  Redis simulation:{sim_id}:broadcast
        (ONE identical message to ALL runners; exclude_runner_ids drops
         already-finished/collapsed runners)
            ↓  dispatcher fan-out to each runner session
runner._runner_before_agent_callback
  └─ if state["velocity"] is None: initialize_runner(...)      # L0, once
  └─ parse JSON, stash into state["_tick_params"]              # bypasses the LLM
            ↓
LLM call  (include_contents="none" — no history, fresh every tick)
  static_instruction : "call process_tick, always, with inner_thought"
  instruction        : "distance: {distance} mi, water: {water}%, velocity:
                        {velocity} mph, status: {runner_status},
                        target finish: {target_finish_minutes} min"
            ↓
process_tick  →  kernel.step(state, env, inner_thought)         # L2
  resource_factor = min_resource_factor + (1−min_resource_factor)·(water/100)
  wall_factor     = 1 − wall_severity   if will_hit_wall and distance > wall_mi
  fatigue_factor  = max(min_fatigue_factor, 1 − natural_fatigue_rate·tick)
  effective_velocity = velocity · resource_factor · wall_factor · fatigue_factor
  depletion       = base_rate · mi · efficiency · (1 + growth·distance)
  auto-refill     at every station marker crossed, seeded RNG per marker
  exhausted / collapsed / arrived  per the scenario's thresholds
            ↓
RPUSH result → collector buffer → simulator drains, aggregates, snapshots
```

Timing: `advance_tick` sleeps `tick_interval_seconds`, draining every 200 ms, and
**early-wakes when all expected runners have reported** — then still waits out the
interval floor. Missing runners are logged, not retried; late finishers are
rescued from stale buffer contents on the next tick to prevent DNF miscounts.

## 4. The finding that matters most: the LLM barely decides anything

The agent README advertises LLM control over pacing intensity and hydration
timing. The **shipped prompt** ([`runner/agent.py`](../../agents/runner/agent.py))
says:

> On every tick event you MUST call `process_tick`. Always. No exceptions.
> Do NOT call any other tool unless explicitly instructed by a non-tick event.
> Do NOT chain multiple tool calls.

Consequences:

- `accelerate`, `brake`, `rehydrate`, `deplete_water`, `get_vitals` are registered
  as tools but are **unreachable during the tick loop** for the LLM runner.
- `crowd_responsiveness` only multiplies `accelerate`, so it is inert for the LLM
  runner — it matters only for `runner_autopilot` plus a frontend `crowd_boost`.
- Hydration decisions were moved *into* the kernel as seeded RNG, explicitly
  removing them from LLM control.
- The LLM's sole causal output is `inner_thought` — a ≤5-word display string.
  Everything else is deterministic given `(session_id, tick)`.

The code comments say why: at hundreds of concurrent runners with `temperature=0.2`
and small models, tool-call *reliability* beat behavioural variety. There is even
a test asserting `inner_thought` has no default, because optional args get dropped
by 2–3B models.

**Read this as the architecture's real thesis: the model owns personality and
legibility; code owns dynamics, safety, and reproducibility.** For evacuation
modelling that split is the right default — but you must *deliberately* re-widen
the tool surface if you want agent choice to be causal.

## 5. Engine vs. marathon-specific

**Reusable engine.** Tick broadcast → per-agent inference → direct-write Redis
collector → aggregate. Seeded profile generation. `include_contents="none"` plus
`static_instruction` (context caching) and dynamic `{var}` state injection, giving
stateless per-tick inference at scale. The **autopilot pattern**: a
`before_model_callback` that returns a synthetic `LlmResponse` containing a
`FunctionCall`, so tools still execute and still emit telemetry while no model is
ever invoked. The two-phase `DECIDE → SUMMARIZE` state machine. Wave staggering.
Gateway spawn → A2A card discovery → sharded Redis spawn queues. Path
parameterisation `t ∈ [0,1]` → 3-D position. The segment-indexed closure model in
[`agents/utils/traffic.py`](../../agents/utils/traffic.py).

**Marathon-specific (now parameterised, see §6).** Every constant in
`runner/constants.py`; `distance` as a scalar on a fixed path; termination as
`distance >= race_distance_mi`; a hydration-only failure model; the
"Spine and Sprout" route planner that builds a *non-self-intersecting closed
26.2-mile loop*; and the frontend's hard-coded `/42195` and `runner_autopilot`
telemetry filter.

**Two limits worth stating plainly:**

- **No agent-to-agent interaction exists.** Every agent receives the identical
  broadcast and reads only its own state. No density, crowding, collision,
  drafting, following, or communication.
- **The frontend runs its own dead-reckoning integrator.** Agents are advanced
  client-side by `effective_velocity`; backend `distance` is a periodic
  correction. The 3-D view is not a pure function of backend state.

## 6. The seams — how this was generalised

| Seam | Change | Status |
|:--|:--|:--|
| **A** Profile generator | distributions move into a scenario spec | ✅ [`spec.py`](../../agents/scenarios/spec.py) |
| **B** Physics kernel | extract a pure `step(state, env, thought)` | ✅ [`kernel.py`](../../agents/runner/kernel.py) |
| **C** Local field | per-segment density in the tick event | 🚧 not built — see below |
| **D** Event vocabulary | widen the enum + `HANDLERS` table | 🚧 partial |
| **E** Position | `path_id + offset` or `(x,y) + heading` | 🚧 not built |
| **F** Termination | goal predicate replaces distance compare | ✅ [`goals.py`](../../agents/scenarios/goals.py) |
| **G** Narrative pooling | reviewed corpus replaces per-agent inference | ✅ [`narrative.py`](../../agents/scenarios/narrative.py) |
| **H** Hazard promotion | perception + salience + admission control | ✅ [`hazard.py`](../../agents/scenarios/hazard.py) |

**Seam C is the highest-leverage remaining work.** Adding a segment-indexed field
to `build_tick_event` — computed from the previous tick's collector drain, which
already contains every agent's position:

```jsonc
{"event":"tick", "tick":42, ...,
 "field":{"segment_len_mi":0.25,
          "density":[0.1,0.4,0.9,...],     // agents per segment
          "hazard":[0,0,0.7,...],
          "blocked":[false,false,true,...]}}
```

Each agent looks up its own segment. This turns independent agents into
interacting ones **without changing the transport, the fan-out, or the
collector**, and it reuses `build_segment_distance_index()`. It is the
prerequisite for queueing, crowd-crush, and rumour propagation.

## 7. Why the marathon is the correctness test

Marathon is deliberately the **degenerate case** of the general model: one
terminal destination, no personas, opportunistic waypoints whose stop/skip
decision remains the kernel's seeded RNG.

That makes it a strict test rather than a coincidence. Because profiles derive
from `sha256(session_id)` and the kernel is pure arithmetic, the entire trajectory
is reproducible bit-for-bit — so
[`agents/tests/test_golden_marathon.py`](../../agents/tests/test_golden_marathon.py)
pins the exact numeric output of 8 runners over 10 ticks. If a refactor changes
one 4th-decimal constant, it fails with the specific runner and tick.

**If the generalised model reproduces marathon byte-for-byte, the abstraction is
right.** That is the entire safety argument for the scenario work.

## 8. Scaling: why per-agent inference does not

The constraint is not primarily cost — it is the **tick barrier**. `advance_tick`
broadcasts, then waits `tick_interval_seconds` (default 10) plus
`max_collection_seconds`. Every runner must complete a full LLM round-trip inside
that window. Late results are drained as **stale** on the next tick and discarded,
so at high N the aggregate silently degrades: `runners_reporting` drops and the
averages are computed over a biased subsample of the fastest responders.

Compounding factors: the barrier is `max()` of N latencies rather than the mean;
the dispatcher fans out with no semaphore; 1000 agents × 12 ticks is 12,000 calls
per 120 s run; and retries make saturation worse. The authors' own verdict is in
the config — `MAX_RUNNERS_AUTOPILOT=100` versus `MAX_RUNNERS_LLM=10`, a deliberate
10× gap.

**The decisive observation is that narrative is already pooled.** Only one thought
is ever visible (the camera-followed agent), and the frontend already ships 65
hand-written fallback strings. At N=1000 the simulation produces ~100 thoughts per
second against a UI that consumes ~0.2 — roughly **500× overproduction of an
output that cannot affect the simulation**. Hence
[`narrative.py`](../../agents/scenarios/narrative.py): a reviewed static corpus
indexed by the same seeded RNG.

Live inference is therefore reserved for the small number of agents promoted by
[`hazard.py`](../../agents/scenarios/hazard.py)'s salience gate — the ones facing
a genuinely causal routing decision.

## 9. Further reading

- [`agents/scenarios/spec.py`](../../agents/scenarios/spec.py) — the frozen
  scenario contract.
- [`agents/runner/README.md`](../../agents/runner/README.md) — the agent's own
  documentation.
- [Agent Architecture](agent_architecture.md) — the ADK/A2A network topology.
- Project [README](../../README.md) §"How the simulation works" — the short
  version of this document.
