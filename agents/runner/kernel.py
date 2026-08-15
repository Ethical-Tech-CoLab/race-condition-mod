# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure per-tick physics kernel for runner agents.

Extracted verbatim from ``running.process_tick`` so the simulation
dynamics can be exercised, tested, and reused **without ADK, Redis, or
any I/O**. ``process_tick`` is now a thin adapter: it resolves tick
parameters, calls :func:`step`, and performs the collector direct-write.

Why this split matters
----------------------
The physics is the authoritative layer -- an LLM policy can bias its
inputs but never override its constraints. Isolating it here means:

* it can be unit-tested with a plain ``dict`` (no ``ToolContext``);
* alternative scenarios can supply different constants/behaviour
  without touching ADK plumbing;
* it can run offline (batch/analysis) where no agent runtime exists.

Contract
--------
:func:`step` **mutates** ``state`` in place (matching the previous
in-place ``tool_context.state`` behaviour) and returns the result
payload. Output is byte-identical to the pre-extraction implementation;
``agents/tests/test_golden_marathon.py`` is the enforcing gate.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from agents.runner.constants import runner_seed
from agents.scenarios.marathon import MARATHON_PHYSICS
from agents.scenarios.spec import PhysicsSpec

logger = logging.getLogger(__name__)

# State is a plain dict (tests/offline) or an ADK State object (runtime).
# Both support ``.get``/``__setitem__``; ``Any`` avoids Pyright conflicts.
StateLike = Any

# Seconds of gateway emission delay applied per wave on the first
# movement tick, so the frontend renders a staggered corral start.
WAVE_STAGGER_SECONDS = 2.0


@dataclass(frozen=True)
class TickEnv:
    """Per-tick environment: everything the kernel needs beyond entity state.

    Immutable so a tick's inputs cannot be mutated mid-step.

    Attributes:
        tick: Current tick number.
        minutes_per_tick: Simulated minutes elapsed this tick.
        elapsed_minutes: Total simulated minutes at end of this tick.
        race_distance_mi: Course length; also the finish predicate bound.
        session_id: Entity identity. Seeds per-hydration-station RNG, so
            drink decisions are reproducible and independent of call
            order. Defaults to ``"default"`` to match the previous
            ``getattr`` fallback chain exactly.
        physics: Scenario dynamics. Defaults to ``MARATHON_PHYSICS`` so
            existing callers are unaffected; other scenarios pass their
            own :class:`PhysicsSpec`.
    """

    tick: int
    minutes_per_tick: float
    elapsed_minutes: float
    race_distance_mi: float
    session_id: str = "default"
    physics: PhysicsSpec = field(default=MARATHON_PHYSICS)


def step(state: StateLike, env: TickEnv, inner_thought: str = "") -> dict:
    """Advance one entity by one tick. Pure physics; mutates ``state``.

    Args:
        state: Entity session state (read and written in place).
        env: Per-tick environment.
        inner_thought: Narrative string passed through to the payload.
            Non-causal -- it never affects dynamics.

    Returns:
        The result payload dict (the same shape ``process_tick`` returns).
    """
    velocity = state.get("velocity", 0.0)
    distance = state.get("distance", 0.0)
    water = state.get("water", 100.0)
    exhausted = state.get("exhausted", False)
    collapsed = state.get("collapsed", False)
    finished = state.get("finished", False)
    phys = env.physics

    # Already finished or collapsed: no-op, but still report so the
    # simulator's aggregation can see this runner (otherwise
    # runners_reporting drops to zero once everyone finishes).
    if finished or collapsed:
        return {
            "status": "success",
            "tick": env.tick,
            "runner_status": "finished" if finished else "collapsed",
            "velocity": velocity,
            "effective_velocity": 0.0,
            "distance_mi": distance,
            "distance": round(distance, 4),
            "water": water,
            "pace_min_per_mi": state.get("pace_min_per_mi"),
            "elapsed_minutes": env.elapsed_minutes,
            "mi_this_tick": 0.0,
            "finish_time_minutes": state.get("finish_time_minutes"),
            "exhausted": exhausted,
            "collapsed": collapsed,
            "inner_thought": inner_thought,
        }

    # --- Effective velocity with degradation factors ---
    # 1. Hydration: min_resource_factor..1.0 of base speed
    hydration_factor = phys.min_resource_factor + (1.0 - phys.min_resource_factor) * (water / 100.0)

    # 2. Wall: sharp pace degradation if past wall_mi
    wall_factor = 1.0
    if state.get("will_hit_wall") and distance > state.get("wall_mi", 18.6411):
        wall_factor = 1.0 - state.get("wall_severity", 0.25)

    # 3. Natural fatigue: gradual slowdown ~0.2% per tick
    fatigue_factor = max(phys.min_fatigue_factor, 1.0 - phys.natural_fatigue_rate * env.tick)

    effective_velocity = velocity * hydration_factor * wall_factor * fatigue_factor

    # --- Distance computation ---
    effective_mph = effective_velocity * phys.speed_scale
    mi_this_tick = effective_mph / 60.0 * env.minutes_per_tick
    raw_distance = distance + mi_this_tick
    new_distance = min(raw_distance, env.race_distance_mi)
    state["distance"] = new_distance

    # --- Hydration depletion ---
    efficiency = state.get("hydration_efficiency", 1.0)
    base_depletion = phys.base_depletion_rate * mi_this_tick * efficiency
    fatigue_growth = 1.0 + phys.fatigue_depletion_growth * new_distance
    depletion = base_depletion * fatigue_growth
    new_water = max(0.0, water - depletion)

    # --- Auto hydration station check (every ~1.86mi) ---
    # Check EVERY station crossed this tick (fast runners may cross 2-3).
    prev_marker = int(distance / phys.resource_station_interval_mi)
    new_marker = int(new_distance / phys.resource_station_interval_mi)
    if new_marker > prev_marker:
        for marker in range(prev_marker + 1, new_marker + 1):
            # Fresh RNG per station for determinism
            rng = random.Random(runner_seed(env.session_id, marker))
            should_drink = (
                new_water <= 40.0
                or exhausted
                or (new_water <= 60.0 and rng.random() < 0.5)
                or (new_water > 60.0 and rng.random() < 0.3)
            )
            if should_drink:
                new_water = min(100.0, new_water + phys.resource_station_refill)

    state["water"] = new_water

    # --- Exhaustion / collapse ---
    if new_water < phys.exhaustion_threshold:
        exhausted = True
    else:
        exhausted = False
    state["exhausted"] = exhausted

    if exhausted and new_water < phys.collapse_threshold:
        collapsed = True
    state["collapsed"] = collapsed

    # --- Finish detection ---
    runner_status = "running"
    finish_time = state.get("finish_time_minutes")
    pace = state.get("pace_min_per_mi")

    if new_distance >= env.race_distance_mi and not finished:
        finished = True
        state["finished"] = True
        runner_status = "finished"
        # Interpolate exact finish time (use raw_distance, before clamping)
        overshoot = raw_distance - env.race_distance_mi
        fraction = 1.0 - (overshoot / mi_this_tick) if mi_this_tick > 0 else 1.0
        finish_time = env.elapsed_minutes - env.minutes_per_tick * (1.0 - fraction)
        pace = finish_time / env.race_distance_mi if env.race_distance_mi > 0 else 0.0
        state["finish_time_minutes"] = round(finish_time, 2)
        state["pace_min_per_mi"] = round(pace, 2)
    elif collapsed:
        runner_status = "collapsed"
    elif exhausted:
        runner_status = "exhausted"

    state["runner_status"] = runner_status

    result = {
        "status": "success",
        "tick": env.tick,
        "runner_status": runner_status,
        "velocity": velocity,
        "effective_velocity": round(effective_velocity, 4),
        "distance_mi": min(round(new_distance, 3), env.race_distance_mi),
        "distance": min(round(new_distance, 4), env.race_distance_mi),
        "water": round(new_water, 1),
        "pace_min_per_mi": pace,
        "elapsed_minutes": env.elapsed_minutes,
        "mi_this_tick": round(mi_this_tick, 3),
        "finish_time_minutes": finish_time,
        "exhausted": exhausted,
        "collapsed": collapsed,
        "inner_thought": inner_thought,
        "wave_number": state.get("wave_number", 0),
    }

    # Stagger the first movement tick's gateway emission by wave so the
    # frontend sees runners start in waves (~2s between each wave).
    wave = state.get("wave_number", 0)
    if env.tick == 1 and wave > 0:
        result["gateway_delay_seconds"] = wave * WAVE_STAGGER_SECONDS

    return result
