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

"""DUMBO, Brooklyn: pedestrian movement toward transit exits.

The first scenario that is *not* a race, and therefore the first real
test of the generalisation. Differences from the marathon:

* **Many ranked exits, not one finish.** Pedestrians head for the
  nearest usable subway entrance -- A/C High St or F York St -- with the
  ferry as a constrained fallback.
* **Exits can close or fill.** Capacity and service windows make the
  ranked fall-through in ``agents/scenarios/goals.py`` load-bearing
  rather than theoretical.
* **Dwell is normal.** Tourists stop at the Washington Street view; that
  is the intended behaviour, not a failure to make progress.
* **Distances are tiny.** DUMBO is roughly half a mile across, versus
  26.2 -- so the hydration mechanic that dominates the marathon is
  deliberately inert here.
* **Personas replace the ability scalar.** A commuter, a tourist, a
  family group and a mobility-limited pedestrian differ categorically,
  not by a single "finish time" number.

Geography is approximate but real: coordinates are ``(lon, lat)`` and
``distance_mi`` is measured along the modelled walking route from the
Brooklyn Bridge Park / Old Fulton Street entry point. Distances are
representative rather than survey-accurate; the planner's GIS tooling
supplies precise geometry when a real route is generated.

The ferry is modelled with the constraints the marathon never needed --
a service window and a hard capacity -- and is deliberately marked
incomplete: real ferry behaviour is a departing vehicle, not a doorway.
See ``FERRY_MODELLING_GAPS``.
"""

from agents.scenarios.spec import (
    Destination,
    DestinationKind,
    PersonaSpec,
    PhysicsSpec,
    ProfileSpec,
    ScenarioSpec,
    TerminationSpec,
)

# --- L2 dynamics ------------------------------------------------------
# Walking, not racing. velocity=1.0 maps to ~3.1 mph, an average adult
# walking pace, so persona speed multipliers read naturally.
#
# The resource ("water") mechanic is intentionally near-inert: nobody
# dehydrates crossing DUMBO. ``min_resource_factor=1.0`` disables the
# speed penalty entirely and the depletion rate is small enough that the
# exhaustion threshold is unreachable over a half-mile walk. This is the
# same kernel with different constants -- no code branches.
WALKING_SPEED_SCALE = 3.1

DUMBO_PHYSICS = PhysicsSpec(
    speed_scale=WALKING_SPEED_SCALE,
    base_depletion_rate=0.05,
    fatigue_depletion_growth=0.0,
    natural_fatigue_rate=0.0,
    min_fatigue_factor=1.0,
    exhaustion_threshold=0.0,
    collapse_threshold=0.0,
    resource_station_interval_mi=999.0,  # effectively no refill points
    resource_station_refill=0.0,
    min_resource_factor=1.0,  # hydration does not affect walking speed
)

# --- L0 identity ------------------------------------------------------
# Retained for schema compatibility; DUMBO drives variation through
# personas rather than a continuous ability draw. The "finish time"
# bounds describe a walk of a few minutes to roughly half an hour
# (a tourist who lingers).
DUMBO_PROFILE = ProfileSpec(
    lognormal_mu=2.3,  # ln(minutes); median ~10 min
    lognormal_sigma=0.45,
    min_finish_min=4.0,
    max_finish_min=35.0,
    wall_hit_probability=0.0,  # no wall when walking half a mile
    course_distance_mi=0.55,
)

# --- L3 world ---------------------------------------------------------

DUMBO_DESTINATIONS: tuple[Destination, ...] = (
    # -- Exits (terminal) ------------------------------------------
    Destination(
        id="subway-high-st",
        kind=DestinationKind.EXIT_SUBWAY,
        name="A/C High St",
        distance_mi=0.30,
        coordinates=(-73.9903, 40.6997),
        preference_rank=0,
        capacity=250,
        # Deep station, stairs and escalator only.
        constraints={"step_free": False, "lines": ("A", "C")},
    ),
    Destination(
        id="subway-york-st",
        kind=DestinationKind.EXIT_SUBWAY,
        name="F York St",
        distance_mi=0.35,
        coordinates=(-73.9868, 40.7014),
        preference_rank=1,
        capacity=200,
        constraints={"step_free": True, "lines": ("F",)},
    ),
    Destination(
        id="ferry-dumbo",
        kind=DestinationKind.EXIT_FERRY,
        name="DUMBO / Fulton Ferry Landing",
        distance_mi=0.22,
        coordinates=(-73.9933, 40.7033),
        preference_rank=2,
        # A boat, not a doorway: small hard capacity, and it only
        # accepts passengers while berthed.
        capacity=149,
        open_ticks=(6, 9),
        constraints={
            "step_free": True,
            "departs_every_ticks": 12,
            "boarding_ticks": 4,
        },
    ),
    # -- Opportunistic stops ---------------------------------------
    Destination(
        id="tour-washington-st",
        kind=DestinationKind.TOUR_STOP,
        name="Washington Street (Manhattan Bridge view)",
        distance_mi=0.12,
        coordinates=(-73.9897, 40.7037),
        preference_rank=0,
    ),
    Destination(
        id="tour-carousel",
        kind=DestinationKind.TOUR_STOP,
        name="Jane's Carousel",
        distance_mi=0.08,
        coordinates=(-73.9956, 40.7033),
        preference_rank=1,
    ),
    Destination(
        id="tour-pebble-beach",
        kind=DestinationKind.TOUR_STOP,
        name="Pebble Beach / Bridge Overlook",
        distance_mi=0.05,
        coordinates=(-73.9948, 40.7040),
        preference_rank=2,
    ),
    Destination(
        id="water-bbp",
        kind=DestinationKind.WATER,
        name="Brooklyn Bridge Park Fountain",
        distance_mi=0.15,
        coordinates=(-73.9941, 40.7025),
    ),
    Destination(
        id="aid-bbp",
        kind=DestinationKind.AID,
        name="Park First Aid Post",
        distance_mi=0.18,
        coordinates=(-73.9930, 40.7020),
    ),
)

# --- Personas ---------------------------------------------------------
# Weights approximate a weekend afternoon: heavy visitor traffic with a
# steady commuter base. Speeds are multipliers on WALKING_SPEED_SCALE.
DUMBO_PERSONAS: tuple[PersonaSpec, ...] = (
    PersonaSpec(
        id="commuter",
        weight=0.30,
        speed_mean=1.15,
        speed_sigma=0.12,
        dwell_probability=0.02,
        # Knows the network; takes the subway and ignores the ferry.
        preferred_kinds=(DestinationKind.EXIT_SUBWAY,),
    ),
    PersonaSpec(
        id="tourist",
        weight=0.40,
        speed_mean=0.80,
        speed_sigma=0.22,
        dwell_probability=0.35,
        # Will happily take the ferry -- it is part of the outing.
        preferred_kinds=(DestinationKind.EXIT_FERRY, DestinationKind.EXIT_SUBWAY),
    ),
    PersonaSpec(
        id="family",
        weight=0.20,
        speed_mean=0.65,
        speed_sigma=0.18,
        dwell_probability=0.30,
        # Strollers need step-free access.
        preferred_kinds=(DestinationKind.EXIT_FERRY, DestinationKind.EXIT_SUBWAY),
        constraints={"step_free": True},
    ),
    PersonaSpec(
        id="mobility_limited",
        weight=0.10,
        speed_mean=0.50,
        speed_sigma=0.10,
        dwell_probability=0.12,
        constraints={"step_free": True},
    ),
)

DUMBO_TERMINATION = TerminationSpec(
    terminal_kinds=frozenset({DestinationKind.EXIT_SUBWAY, DestinationKind.EXIT_FERRY}),
    max_ticks=60,
    end_when_all_terminal=True,
)

DUMBO = ScenarioSpec(
    id="dumbo",
    name="DUMBO, Brooklyn",
    description=(
        "Pedestrian movement through DUMBO toward transit exits. Visitors "
        "and commuters head for A/C High St or F York St, with the East "
        "River Ferry as a capacity- and schedule-constrained alternative. "
        "Exercises ranked destinations, closure fall-through, persona "
        "accessibility constraints, and dwell behaviour."
    ),
    physics=DUMBO_PHYSICS,
    profile=DUMBO_PROFILE,
    termination=DUMBO_TERMINATION,
    destinations=DUMBO_DESTINATIONS,
    personas=DUMBO_PERSONAS,
)

# Known gaps, recorded so the ferry is not mistaken for a finished model.
# A ferry is a *vehicle*, not a doorway: passengers accumulate on the
# pier, board in a burst, and depart together, so arrivals are batched
# rather than continuous. Modelling that properly needs the queueing and
# capacity work in later phases.
FERRY_MODELLING_GAPS = (
    "capacity is declared but not yet enforced (no queueing)",
    "boarding is instantaneous; real boarding is a batch at departure",
    "a single service window is modelled, not a repeating timetable",
    "missing the boat should return the pedestrian to the ranked exits",
    "weather and river conditions can suspend service entirely",
)
