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

"""Urban evacuation under hazard: a Mariupol-shaped scenario.

The third reference scenario, and the one that exercises the parts the
marathon never needed: ranked exits that close under threat, shelters as
an alternative to leaving, and agents whose attention is driven by
events rather than by a clock.

Modelling stance
----------------
This scenario is **schematic**. It is shaped by the documented pattern
of a besieged city -- humanitarian corridors that open and close at
short notice, shelters of varying protection, aid points, and civilians
who differ sharply in mobility -- but it is not a reconstruction of
specific events, and the numbers here are illustrative parameters, not
findings.

Ethical boundary (enforced by design, not convention)
-----------------------------------------------------
* Hazards influence **routing and attention only**. Attrition,
  casualties, and mortality are deliberately **not** modelled.
* All dynamics stay in deterministic, seeded, reviewable code, so any
  run is reproducible and auditable.
* Narrative, when added, must come from a **reviewed static corpus**,
  never live generation. A language model must not improvise about a
  real siege with real casualties.

These are the reasons the deterministic skeleton is built first and the
LLM tier last: the parts that could cause harm are the parts that must
be inspectable.

Geometry is 1-D along an evacuation route, consistent with the kernel's
``distance_mi`` coordinate. ``coordinates`` are approximate real
positions in and around Mariupol for map rendering only.
"""

from agents.scenarios.hazard import Hazard, HazardKind
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
# Movement on foot, carrying belongings, often with children or elderly
# relatives: slower than ordinary walking. velocity=1.0 maps to ~2.4 mph.
#
# The resource mechanic is repurposed from hydration to **stamina** and
# is meaningful here: this is a long walk under stress, unlike DUMBO.
# Exhaustion slows people down; it does not kill them. The collapse
# threshold is set to 0 so the kernel can never mark anyone "collapsed"
# in this scenario -- that outcome is out of scope by design.
EVACUATION_SPEED_SCALE = 2.4

MARIUPOL_PHYSICS = PhysicsSpec(
    speed_scale=EVACUATION_SPEED_SCALE,
    base_depletion_rate=1.4,
    fatigue_depletion_growth=0.02,
    natural_fatigue_rate=0.001,
    min_fatigue_factor=0.70,
    exhaustion_threshold=25.0,
    # Never collapse: attrition is out of scope for this scenario.
    collapse_threshold=0.0,
    resource_station_interval_mi=3.0,
    resource_station_refill=20.0,
    # Exhaustion slows movement to 60% rather than halving it.
    min_resource_factor=0.6,
)

MARIUPOL_PROFILE = ProfileSpec(
    lognormal_mu=4.6,  # ln(minutes); median ~100 min on foot
    lognormal_sigma=0.40,
    min_finish_min=45.0,
    max_finish_min=420.0,
    wall_hit_probability=0.0,  # fatigue is modelled directly, not as a "wall"
    course_distance_mi=7.5,
)

# --- L3 world ---------------------------------------------------------
# Two corridor exits (the intended way out), two shelters (the choice to
# stay), and aid/water along the route.
MARIUPOL_DESTINATIONS: tuple[Destination, ...] = (
    # -- Corridor exits: preferred, but fragile -----------------------
    Destination(
        id="corridor-west",
        kind=DestinationKind.CORRIDOR_EXIT,
        name="Western Humanitarian Corridor",
        distance_mi=7.5,
        coordinates=(37.4630, 47.0951),
        preference_rank=0,
        capacity=4000,
        # Corridors are announced for a window and frequently withdrawn.
        open_ticks=(0, 40),
        constraints={"vehicle_assisted": True, "step_free": True},
    ),
    Destination(
        id="corridor-north",
        kind=DestinationKind.CORRIDOR_EXIT,
        name="Northern Assembly Point",
        distance_mi=5.8,
        coordinates=(37.5490, 47.1440),
        preference_rank=1,
        capacity=2500,
        open_ticks=(0, 28),
        constraints={"step_free": True},
    ),
    # -- Shelters: the alternative to leaving -------------------------
    Destination(
        id="shelter-theatre",
        kind=DestinationKind.SHELTER,
        name="Central Shelter",
        distance_mi=1.2,
        coordinates=(37.5490, 47.0958),
        preference_rank=2,
        capacity=1200,
        constraints={"step_free": True, "below_ground": True},
    ),
    Destination(
        id="shelter-school",
        kind=DestinationKind.SHELTER,
        name="School Basement Shelter",
        distance_mi=3.1,
        coordinates=(37.5720, 47.1090),
        preference_rank=3,
        capacity=400,
        constraints={"step_free": False, "below_ground": True},
    ),
    # -- Aid and water along the route --------------------------------
    Destination(
        id="aid-central",
        kind=DestinationKind.AID,
        name="Central Aid Point",
        distance_mi=2.0,
        coordinates=(37.5560, 47.1010),
        preference_rank=0,
    ),
    Destination(
        id="aid-north",
        kind=DestinationKind.AID,
        name="Northern Aid Point",
        distance_mi=4.6,
        coordinates=(37.5600, 47.1300),
        preference_rank=1,
    ),
    Destination(
        id="water-central",
        kind=DestinationKind.WATER,
        name="Water Distribution Point",
        distance_mi=2.6,
        coordinates=(37.5530, 47.1050),
    ),
    Destination(
        id="water-north",
        kind=DestinationKind.WATER,
        name="Northern Water Point",
        distance_mi=5.2,
        coordinates=(37.5580, 47.1360),
    ),
)

# --- Personas ---------------------------------------------------------
# Households, not individuals: people evacuate in groups and move at the
# pace of their slowest member. ``group_size`` and ``stakes`` are carried
# in constraints for use by later phases (cohesion, salience weighting).
MARIUPOL_PERSONAS: tuple[PersonaSpec, ...] = (
    PersonaSpec(
        id="family_with_children",
        weight=0.30,
        speed_mean=0.70,
        speed_sigma=0.15,
        dwell_probability=0.10,
        preferred_kinds=(DestinationKind.CORRIDOR_EXIT, DestinationKind.SHELTER),
        constraints={"step_free": True},
        attributes={"group_size": 4, "stakes": 0.9},
    ),
    PersonaSpec(
        id="adult_alone",
        weight=0.30,
        speed_mean=1.10,
        speed_sigma=0.18,
        dwell_probability=0.03,
        preferred_kinds=(DestinationKind.CORRIDOR_EXIT, DestinationKind.SHELTER),
        attributes={"group_size": 1, "stakes": 0.4},
    ),
    PersonaSpec(
        id="elderly",
        weight=0.25,
        speed_mean=0.50,
        speed_sigma=0.12,
        dwell_probability=0.15,
        # More likely to shelter in place than attempt a long walk out.
        preferred_kinds=(DestinationKind.SHELTER, DestinationKind.CORRIDOR_EXIT),
        constraints={"step_free": True},
        attributes={"group_size": 2, "stakes": 1.0},
    ),
    PersonaSpec(
        id="mobility_limited",
        weight=0.15,
        speed_mean=0.35,
        speed_sigma=0.08,
        dwell_probability=0.20,
        preferred_kinds=(DestinationKind.SHELTER,),
        # Only step-free access is a hard requirement. Vehicle
        # assistance is a need for *long* journeys, not a property every
        # destination must advertise, so it lives in attributes.
        constraints={"step_free": True},
        attributes={"group_size": 2, "stakes": 1.0, "needs_vehicle_for_distance": True},
    ),
)

MARIUPOL_TERMINATION = TerminationSpec(
    terminal_kinds=frozenset({DestinationKind.CORRIDOR_EXIT, DestinationKind.SHELTER}),
    max_ticks=120,
    end_when_all_terminal=True,
)

# --- Hazards ----------------------------------------------------------
# Illustrative, not a reconstruction. Chosen to exercise distinct
# perception and closure behaviours:
#   * shelling      -- sudden, brief warning, static blast radius;
#   * troop movement -- slow, long warning, steadily advancing;
#   * fire          -- gradual, wind-driven spread from a strike.
MARIUPOL_HAZARDS: tuple[Hazard, ...] = (
    Hazard(
        id="shelling-centre",
        kind=HazardKind.SHELLING,
        onset_tick=6,
        origin_mi=2.2,
        severity=0.85,
        initial_radius_mi=0.4,
        growth_mi_per_tick=0.0,
        warning_ticks=1,  # air-raid siren, minimal lead time
        blocks_travel=True,
    ),
    Hazard(
        id="advance-north",
        kind=HazardKind.TROOP_MOVEMENT,
        onset_tick=14,
        origin_mi=5.8,
        severity=0.7,
        initial_radius_mi=0.5,
        growth_mi_per_tick=0.12,  # steadily encroaching
        warning_ticks=8,  # widely known in advance
        blocks_travel=True,
    ),
    Hazard(
        id="fire-industrial",
        kind=HazardKind.FIRE,
        onset_tick=20,
        origin_mi=4.0,
        severity=0.6,
        initial_radius_mi=0.2,
        growth_mi_per_tick=0.08,
        warning_ticks=2,
        blocks_travel=True,
    ),
)

MARIUPOL = ScenarioSpec(
    id="mariupol",
    name="Urban Evacuation (Mariupol-shaped)",
    description=(
        "Schematic urban evacuation under hazard. Civilians choose between "
        "humanitarian corridors that close at short notice and shelters "
        "that mean staying. Exercises hazard perception, ranked-exit "
        "fall-through, accessibility constraints, and event-driven "
        "attention. Attrition and casualties are deliberately not "
        "modelled; hazards affect routing and attention only."
    ),
    physics=MARIUPOL_PHYSICS,
    profile=MARIUPOL_PROFILE,
    termination=MARIUPOL_TERMINATION,
    destinations=MARIUPOL_DESTINATIONS,
    personas=MARIUPOL_PERSONAS,
)

# Explicit statement of what this scenario does not attempt, so the
# model's limits travel with it.
MODELLING_LIMITS = (
    "attrition, casualties and mortality are out of scope by design",
    "household groups are declared but cohesion is not yet simulated",
    "information/rumour propagation needs the density field",
    "corridor availability is scripted, not negotiated",
    "vehicle-assisted evacuation is a constraint flag, not a transport model",
    "geometry is 1-D along a route, not a street network",
)
