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

"""Goal selection and arrival, generalising the marathon finish line.

The marathon hard-codes termination as ``distance >= race_distance_mi``:
one destination, identical for everyone, never unavailable. Every other
scenario needs agents to *choose* among ranked destinations that can
close underneath them.

This module supplies that choice while keeping the marathon a strict
special case:

* marathon -> a single ``FINISH``; selection is trivial and arrival
  reduces to the original distance comparison;
* Dumbo -> ranked subway entrances (A/High St, then F/York St) with a
  capacity- and schedule-constrained ferry behind them;
* evacuation -> ranked corridor exits and shelters that hazards close.

Selection is **deterministic**: given the same spec, tick, persona and
excluded set, the same destination is returned. No RNG is used here, so
goal choice cannot perturb the seeded trajectories the golden fixture
pins.

Only *terminal* destinations are goals. Opportunistic stops (water, aid,
tour stops) are waypoints handled by the kernel's own logic and do not
end a run.
"""

from agents.scenarios.spec import Destination, DestinationKind, PersonaSpec, ScenarioSpec


def select_goal(
    spec: ScenarioSpec,
    tick: int = 0,
    persona: PersonaSpec | None = None,
    exclude_ids: frozenset[str] = frozenset(),
) -> Destination | None:
    """Pick the best currently-available terminal destination.

    Ordering rules, in priority order:

    1. destination must be open at ``tick`` (closures remove options);
    2. destination must not be in ``exclude_ids`` (already-rejected, e.g.
       observed full or unreachable);
    3. if ``persona`` declares ``preferred_kinds``, those kinds sort
       first, in the persona's stated order;
    4. otherwise the scenario's ``preference_rank`` applies, ties broken
       by ``id`` for stability.

    Args:
        spec: The scenario being run.
        tick: Current tick, used to evaluate opening windows.
        persona: Optional archetype whose preferences override the
            scenario ranking (e.g. mobility-limited agents avoiding
            stair-only entrances).
        exclude_ids: Destination ids this agent has ruled out.

    Returns:
        The chosen destination, or ``None`` if every terminal
        destination is closed or excluded -- a meaningful state in an
        evacuation (no way out) that callers must handle.
    """
    candidates = [d for d in spec.open_terminal_destinations(tick) if d.id not in exclude_ids]
    if persona is not None:
        candidates = [d for d in candidates if _satisfies(d, persona)]
    if not candidates:
        return None

    if persona is not None and persona.preferred_kinds:
        order = {kind: i for i, kind in enumerate(persona.preferred_kinds)}
        # Unlisted kinds sort after every listed one.
        fallback = len(order)
        candidates.sort(key=lambda d: (order.get(d.kind, fallback), d.preference_rank, d.id))
    else:
        candidates.sort(key=lambda d: (d.preference_rank, d.id))

    return candidates[0]


def _satisfies(destination: Destination, persona: PersonaSpec) -> bool:
    """Whether ``destination`` meets every constraint the persona requires.

    A persona constraint is a hard requirement: if it asks for
    ``{"step_free": True}``, a destination must explicitly offer
    ``step_free`` truthy. Absent means not offered, so not eligible.
    """
    for key, required in persona.constraints.items():
        if destination.constraints.get(key) != required:
            return False
    return True


def goal_distance_mi(goal: Destination | None, default_mi: float) -> float:
    """Distance at which ``goal`` is reached, along the 1-D course.

    Falls back to ``default_mi`` when there is no goal or the goal has
    no position, which is what keeps marathon behaviour identical.
    """
    if goal is None or goal.distance_mi is None:
        return default_mi
    return goal.distance_mi


def has_arrived(distance_mi: float, goal: Destination | None, default_mi: float) -> bool:
    """Whether an agent at ``distance_mi`` has reached its goal.

    Generalises ``distance >= race_distance_mi``; for the marathon the
    two are the same comparison.
    """
    return distance_mi >= goal_distance_mi(goal, default_mi)


def next_goal_after_closure(
    spec: ScenarioSpec,
    current: Destination | None,
    tick: int,
    persona: PersonaSpec | None = None,
    exclude_ids: frozenset[str] = frozenset(),
) -> Destination | None:
    """Re-select when the current goal has become unavailable.

    Returns ``current`` unchanged while it is still open and allowed, so
    callers can invoke this every tick cheaply. When the preferred exit
    shuts, this is what falls the agent through to the next rank.
    """
    if current is not None and current.id not in exclude_ids and current.is_open_at(tick):
        if persona is None or _satisfies(current, persona):
            return current

    excluded = exclude_ids | ({current.id} if current is not None else set())
    return select_goal(spec, tick=tick, persona=persona, exclude_ids=excluded)


def opportunistic_destinations(
    spec: ScenarioSpec,
    kind: DestinationKind,
    tick: int = 0,
) -> tuple[Destination, ...]:
    """Open, non-terminal destinations of ``kind``, in course order.

    Waypoints (water, aid, tour stops) are consumed en route rather than
    chosen as goals, so they are ordered by position rather than
    preference.
    """
    return tuple(
        sorted(
            (d for d in spec.destinations_of(kind) if d.is_open_at(tick)),
            key=lambda d: (d.distance_mi if d.distance_mi is not None else 0.0, d.id),
        )
    )
