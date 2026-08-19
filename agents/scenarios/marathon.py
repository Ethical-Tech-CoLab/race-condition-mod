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

"""Marathon expressed as a :class:`ScenarioSpec`.

This is the reference scenario and the regression anchor: every value
here is imported from ``agents/runner/constants.py`` rather than
re-typed, so the constants remain the single source of truth and this
module cannot silently drift from them.

Marathon is deliberately the *degenerate* case of the general model:

* exactly one terminal destination (the finish line);
* opportunistic ``WATER`` waypoints whose stop/skip decision stays with
  the kernel's seeded per-station RNG (unchanged behaviour);
* a single implicit persona -- ability is drawn as a continuous scalar
  rather than sampled from categories.

If this spec ever fails to reproduce the golden fixture, the
generalisation is wrong.
"""

from agents.runner.constants import (
    BASE_DEPLETION_RATE,
    COLLAPSE_THRESHOLD,
    EXHAUSTION_THRESHOLD,
    FATIGUE_DEPLETION_GROWTH,
    HYDRATION_STATION_INTERVAL_MI,
    HYDRATION_STATION_REFILL,
    LOGNORMAL_MU,
    LOGNORMAL_SIGMA,
    MARATHON_MI,
    MAX_FINISH_MIN,
    MIN_FATIGUE_FACTOR,
    MIN_FINISH_MIN,
    NATURAL_FATIGUE_RATE,
    SPEED_SCALE,
    WALL_HIT_PROBABILITY,
)
from agents.scenarios.spec import (
    Destination,
    DestinationKind,
    PhysicsSpec,
    ProfileSpec,
    ScenarioSpec,
    TerminationSpec,
)

# --- L2 dynamics -----------------------------------------------------
# Mirrors the constants the kernel used directly before Phase 2.
MARATHON_PHYSICS = PhysicsSpec(
    speed_scale=SPEED_SCALE,
    base_depletion_rate=BASE_DEPLETION_RATE,
    fatigue_depletion_growth=FATIGUE_DEPLETION_GROWTH,
    natural_fatigue_rate=NATURAL_FATIGUE_RATE,
    min_fatigue_factor=MIN_FATIGUE_FACTOR,
    exhaustion_threshold=EXHAUSTION_THRESHOLD,
    collapse_threshold=COLLAPSE_THRESHOLD,
    resource_station_interval_mi=HYDRATION_STATION_INTERVAL_MI,
    resource_station_refill=HYDRATION_STATION_REFILL,
    min_resource_factor=0.5,
)

# --- L0 identity distributions ---------------------------------------
MARATHON_PROFILE = ProfileSpec(
    lognormal_mu=LOGNORMAL_MU,
    lognormal_sigma=LOGNORMAL_SIGMA,
    min_finish_min=MIN_FINISH_MIN,
    max_finish_min=MAX_FINISH_MIN,
    wall_hit_probability=WALL_HIT_PROBABILITY,
    course_distance_mi=MARATHON_MI,
)

# --- L3 world --------------------------------------------------------
# Water stations are generated at the same ~1.86 mi cadence the kernel
# already checks, so the declarative view matches the computed one.
# ``int(MARATHON_MI / interval)`` mirrors the kernel's marker maths.
_STATION_COUNT = int(MARATHON_MI / HYDRATION_STATION_INTERVAL_MI)

MARATHON_DESTINATIONS: tuple[Destination, ...] = (
    Destination(
        id="finish",
        kind=DestinationKind.FINISH,
        name="Finish Line",
        distance_mi=MARATHON_MI,
        preference_rank=0,
    ),
    *(
        Destination(
            id=f"water-{i}",
            kind=DestinationKind.WATER,
            name=f"Hydration Station {i}",
            distance_mi=round(i * HYDRATION_STATION_INTERVAL_MI, 4),
            preference_rank=i,
        )
        for i in range(1, _STATION_COUNT + 1)
    ),
    # Medical tents: halfway and finish, matching the planner's
    # ``_place_medical_stations`` placement.
    Destination(
        id="aid-half",
        kind=DestinationKind.AID,
        name="Medical Tent (Halfway)",
        distance_mi=round(MARATHON_MI / 2, 4),
        preference_rank=0,
    ),
    Destination(
        id="aid-finish",
        kind=DestinationKind.AID,
        name="Medical Tent (Finish)",
        distance_mi=MARATHON_MI,
        preference_rank=1,
    ),
)

# --- Termination -----------------------------------------------------
# Only FINISH is terminal here: a marathon runner does not "exit" at a
# water station or a medical tent.
MARATHON_TERMINATION = TerminationSpec(
    terminal_kinds=frozenset({DestinationKind.FINISH}),
    max_ticks=200,
    end_when_all_terminal=True,
)

MARATHON = ScenarioSpec(
    id="marathon",
    name="Las Vegas Marathon",
    description=(
        "The reference scenario: a 26.2 mile point-to-point race with a "
        "single finish line and opportunistic hydration. Serves as the "
        "regression anchor for scenario generalisation."
    ),
    physics=MARATHON_PHYSICS,
    profile=MARATHON_PROFILE,
    termination=MARATHON_TERMINATION,
    destinations=MARATHON_DESTINATIONS,
    # No personas: marathon ability is a continuous scalar drawn from
    # ``ProfileSpec``, not a categorical archetype.
    personas=(),
)
