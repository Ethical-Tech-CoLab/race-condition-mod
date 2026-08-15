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

"""Scenario schema: the contract shared by every simulated event.

This is the generalization of the marathon simulation into a
*scenario-parameterised* one, so the same engine can run a marathon, a
pedestrian district, or an evacuation.

Design premise
--------------
The marathon is a **degenerate case** of the general model: one terminal
destination (``FINISH``) plus opportunistic waypoints (``WATER``). If the
generalised schema reproduces marathon output byte-for-byte, the
abstraction is correct. ``agents/tests/test_golden_marathon.py`` enforces
exactly that.

The unifying abstraction is the **destination**. A marathon water stop, a
Dumbo subway entrance, an evacuation shelter, and an aid post are the
same kind of thing: a typed point on the course with optional capacity,
opening hours, and a preference rank. The repo already *places* such
points (``_place_hydration_stations`` and friends in the planner) but
treats them as inert map markers; this schema makes them addressable
goals.

Layering
--------
Mirrors the four-layer decomposition of a runner:

* ``ProfileSpec``  -> L0 identity (who this agent is; seeded)
* ``PersonaSpec``  -> L0 identity, categorical variant
* ``PhysicsSpec``  -> L2 dynamics (constants the kernel enforces)
* ``Destination``  -> L3 world (where agents can go)
* ``TerminationSpec`` -> when an agent is done

``PhysicsSpec`` is the only part wired into the kernel today. Personas,
destinations, and termination are defined here so scenario packs can be
authored against a frozen schema; they are wired into behaviour in
subsequent phases.

All specs are frozen dataclasses: a scenario is configuration, and must
not mutate mid-run.
"""

from dataclasses import dataclass, field
from enum import Enum


class DestinationKind(str, Enum):
    """Typed destinations, shared across scenarios.

    ``str`` mixin so values serialise directly into tick events and
    telemetry without custom encoders.
    """

    # Terminal -- reaching one ends the agent's run.
    FINISH = "finish"  # marathon finish line
    EXIT_SUBWAY = "exit_subway"  # Dumbo: A/High St, F/York St
    EXIT_FERRY = "exit_ferry"  # Dumbo: capacity + schedule constrained
    CORRIDOR_EXIT = "corridor_exit"  # evacuation: humanitarian corridor
    SHELTER = "shelter"  # evacuation: in-place refuge

    # Opportunistic -- visited en route, does not end the run.
    WATER = "water"  # marathon hydration station
    AID = "aid"  # medical/triage, all scenarios
    TOUR_STOP = "tour_stop"  # Dumbo: photo spot / dwell POI

    @property
    def is_terminal(self) -> bool:
        """True if arriving here ends the agent's run."""
        return self in _TERMINAL_KINDS


_TERMINAL_KINDS = frozenset(
    {
        DestinationKind.FINISH,
        DestinationKind.EXIT_SUBWAY,
        DestinationKind.EXIT_FERRY,
        DestinationKind.CORRIDOR_EXIT,
        DestinationKind.SHELTER,
    }
)


@dataclass(frozen=True)
class Destination:
    """A typed, optionally-constrained place an agent can head for.

    Attributes:
        id: Stable identifier, unique within a scenario.
        kind: Semantic type; drives goal selection and arrival handling.
        name: Human-readable label for UI and narrative.
        distance_mi: Position along the course, in miles from the start.
            Matches the kernel's 1-D ``distance`` model. Scenarios needing
            true 2-D placement carry ``coordinates`` as well.
        coordinates: Optional ``(lon, lat)`` for map rendering.
        preference_rank: Lower is preferred. Lets a scenario express
            "A/High St first, else F/York St, else the ferry."
        capacity: Max agents that can be served; ``None`` is unbounded.
            Drives queueing once density is modelled.
        open_ticks: Optional inclusive ``(first, last)`` tick window.
            ``None`` means always open. Closure is how hazards and
            ferry schedules remove options.
        constraints: Free-form scenario-specific requirements, e.g.
            ``{"step_free": True}`` or ``{"departs_every_ticks": 12}``.
    """

    id: str
    kind: DestinationKind
    name: str = ""
    distance_mi: float | None = None
    coordinates: tuple[float, float] | None = None
    preference_rank: int = 0
    capacity: int | None = None
    open_ticks: tuple[int, int] | None = None
    constraints: dict = field(default_factory=dict)

    def is_open_at(self, tick: int) -> bool:
        """Whether this destination accepts arrivals at ``tick``."""
        if self.open_ticks is None:
            return True
        first, last = self.open_ticks
        return first <= tick <= last


@dataclass(frozen=True)
class PhysicsSpec:
    """L2 constants the kernel enforces regardless of agent decisions.

    Extracted from ``agents/runner/constants.py`` so alternative
    scenarios can supply their own dynamics without editing the kernel.
    Field names mirror the original constants to keep the mapping
    reviewable.

    Naming note: the marathon calls the depleting quantity "water", but
    the mechanic is generic (stamina, fuel, battery), so the spec uses
    "resource".
    """

    speed_scale: float
    base_depletion_rate: float
    fatigue_depletion_growth: float
    natural_fatigue_rate: float
    min_fatigue_factor: float
    exhaustion_threshold: float
    collapse_threshold: float
    resource_station_interval_mi: float
    resource_station_refill: float
    # Speed retained at zero resource. The kernel computes
    # ``min_resource_factor + (1 - min_resource_factor) * (r / 100)``,
    # so 0.5 reproduces the marathon's "half speed when empty".
    min_resource_factor: float = 0.5


@dataclass(frozen=True)
class ProfileSpec:
    """L0 seeded distributions that generate an agent's identity.

    Marathon draws a single continuous ability scalar
    (``target_finish_minutes``) and derives everything from it. Other
    scenarios may lean on ``personas`` instead; both can coexist.
    """

    lognormal_mu: float
    lognormal_sigma: float
    min_finish_min: float
    max_finish_min: float
    wall_hit_probability: float
    course_distance_mi: float


@dataclass(frozen=True)
class PersonaSpec:
    """A categorical agent archetype.

    Behaviour within a persona is randomised but "substantially the
    same", which is what makes persona-level batching of narrative (and
    later, of decisions) sound: members of a cohort face materially the
    same situation.

    Attributes:
        id: Stable key; also the lookup key for narrative pools.
        weight: Relative sampling weight; need not sum to 1.
        speed_mean / speed_sigma: Gaussian base speed, in the kernel's
            normalised velocity units.
        dwell_probability: Chance of pausing at a ``TOUR_STOP``.
        preferred_kinds: Destination kinds this persona favours, in
            order, overriding the global ranking.
        constraints: Requirements a destination must satisfy, e.g.
            ``{"step_free": True}`` for mobility-limited agents.
    """

    id: str
    weight: float = 1.0
    speed_mean: float = 1.0
    speed_sigma: float = 0.15
    dwell_probability: float = 0.0
    preferred_kinds: tuple[DestinationKind, ...] = ()
    constraints: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TerminationSpec:
    """When an agent's run ends, and when the whole scenario ends.

    Generalises the marathon's hard-coded
    ``distance >= race_distance_mi``.

    Attributes:
        terminal_kinds: Arriving at any of these ends the agent's run.
        max_ticks: Scenario-level tick budget.
        end_when_all_terminal: Stop early once every agent is done.
    """

    terminal_kinds: frozenset = _TERMINAL_KINDS
    max_ticks: int = 200
    end_when_all_terminal: bool = True


@dataclass(frozen=True)
class ScenarioSpec:
    """A complete, self-contained scenario definition.

    This is the object a scenario pack exports and the simulator
    consumes. Freezing this schema is what allows scenario packs to be
    authored in parallel: packs own disjoint files and share only this
    contract.
    """

    id: str
    name: str
    physics: PhysicsSpec
    profile: ProfileSpec
    termination: TerminationSpec
    destinations: tuple[Destination, ...] = ()
    personas: tuple[PersonaSpec, ...] = ()
    description: str = ""

    def destinations_of(self, kind: DestinationKind) -> tuple[Destination, ...]:
        """All destinations of ``kind``, ordered by preference then id."""
        return tuple(
            sorted(
                (d for d in self.destinations if d.kind == kind),
                key=lambda d: (d.preference_rank, d.id),
            )
        )

    def terminal_destinations(self) -> tuple[Destination, ...]:
        """Run-ending destinations, most preferred first.

        This is the ranked exit list: for Dumbo, A/High St then
        F/York St then the ferry.
        """
        return tuple(
            sorted(
                (d for d in self.destinations if d.kind in self.termination.terminal_kinds),
                key=lambda d: (d.preference_rank, d.id),
            )
        )

    def open_terminal_destinations(self, tick: int) -> tuple[Destination, ...]:
        """Terminal destinations still open at ``tick``.

        Closure is how hazards and schedules remove options: when the
        preferred exit shuts, agents fall through to the next rank.
        """
        return tuple(d for d in self.terminal_destinations() if d.is_open_at(tick))
