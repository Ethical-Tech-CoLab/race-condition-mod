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

"""Hazards, perception, and salience -- event-driven agent attention.

Corrects a simplification in the earlier design: "an agent on a straight
segment has no decision to make". That is only true until something
happens *to* it. Decisions are triggered by a change in the agent's
**information state**, not by road topology -- and an inbound hazard can
arrive at any point in time and space.

Three separable layers, all deterministic:

* **H1 hazard field** -- what is objectively happening (:class:`Hazard`).
* **H2 perception** -- what *this* agent knows (:func:`perceives`).
  Agents respond to sirens, sightlines, broadcast alerts and each other,
  never to ground truth directly. This is the layer most simulations
  omit, and it is where evacuation realism lives.
* **H3 salience** -- whether the agent needs to *think*
  (:func:`compute_salience`). Feeds the promotion gate that decides
  between cheap deterministic movement and expensive reasoning.

Everything here is seeded from ``(session_id, tick, hazard id)`` so runs
stay reproducible and auditable. No global RNG is used.

**Deliberate boundary.** Attrition, casualty, and mortality outcomes are
*not* modelled here and must stay in deterministic, seeded, reviewable
code. For the evacuation scenario this is an ethical requirement, not a
stylistic one: a language model must never improvise about who lives or
dies. Hazards here affect *routing and attention* only.
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum

from agents.runner.constants import runner_seed


class HazardKind(str, Enum):
    """Hazard archetypes, distinguished by onset, warning and spread."""

    SHELLING = "shelling"  # instant onset, seconds of warning, static blast
    TROOP_MOVEMENT = "troop_movement"  # slow, directional, hours of warning
    FLOOD = "flood"  # gradual, terrain-following
    FIRE = "fire"  # gradual, wind-driven front
    TSUNAMI = "tsunami"  # sudden, minutes of warning, coast-following
    EARTHQUAKE = "earthquake"  # instant, radial, aftershocks
    STRUCTURAL = "structural"  # building/bridge collapse, static
    CROWD_CRUSH = "crowd_crush"  # density-induced, emergent


@dataclass(frozen=True)
class Hazard:
    """A threat with a position, a severity, and a growth profile.

    Positions use the same 1-D ``distance_mi`` course coordinate the
    kernel uses, so hazards can be reasoned about without 2-D geometry.

    Attributes:
        id: Stable identifier.
        kind: Archetype; drives perception ranges and defaults.
        onset_tick: Tick at which the hazard becomes real.
        origin_mi: Position along the course.
        severity: 0..1 intensity at the origin.
        initial_radius_mi: Affected radius at onset.
        growth_mi_per_tick: Radius growth per tick (0 = static).
        warning_ticks: Lead time before onset during which a broadcast
            alert can reach equipped agents (sirens, phone alerts, EEW).
        blocks_travel: Whether the affected span is impassable, which
            closes destinations inside it.
    """

    id: str
    kind: HazardKind
    onset_tick: int
    origin_mi: float
    severity: float = 1.0
    initial_radius_mi: float = 0.1
    growth_mi_per_tick: float = 0.0
    warning_ticks: int = 0
    blocks_travel: bool = True

    def is_active_at(self, tick: int) -> bool:
        """Whether the hazard has begun by ``tick``."""
        return tick >= self.onset_tick

    def radius_at(self, tick: int) -> float:
        """Affected radius at ``tick``; zero before onset."""
        if not self.is_active_at(tick):
            return 0.0
        return self.initial_radius_mi + self.growth_mi_per_tick * (tick - self.onset_tick)

    def affects(self, distance_mi: float, tick: int) -> bool:
        """Whether a point on the course is inside the hazard."""
        if not self.is_active_at(tick):
            return False
        return abs(distance_mi - self.origin_mi) <= self.radius_at(tick)

    def intensity_at(self, distance_mi: float, tick: int) -> float:
        """Severity experienced at a point, decaying linearly to the edge."""
        if not self.affects(distance_mi, tick):
            return 0.0
        radius = self.radius_at(tick)
        if radius <= 0:
            return self.severity
        proximity = 1.0 - (abs(distance_mi - self.origin_mi) / radius)
        return self.severity * max(0.0, proximity)

    def ticks_until_onset(self, tick: int) -> int:
        """Ticks remaining before onset; 0 once active."""
        return max(0, self.onset_tick - tick)


# Perception ranges in miles, by hazard kind: how far it can be seen and
# heard. Explosions carry much further than a rising flood.
_SIGHT_MI: dict[HazardKind, float] = {
    HazardKind.SHELLING: 1.5,
    HazardKind.TROOP_MOVEMENT: 0.8,
    HazardKind.FLOOD: 0.4,
    HazardKind.FIRE: 1.2,
    HazardKind.TSUNAMI: 2.0,
    HazardKind.EARTHQUAKE: 5.0,
    HazardKind.STRUCTURAL: 0.5,
    HazardKind.CROWD_CRUSH: 0.15,
}
_SOUND_MI: dict[HazardKind, float] = {
    HazardKind.SHELLING: 4.0,
    HazardKind.TROOP_MOVEMENT: 1.5,
    HazardKind.FLOOD: 0.5,
    HazardKind.FIRE: 0.8,
    HazardKind.TSUNAMI: 1.5,
    HazardKind.EARTHQUAKE: 6.0,
    HazardKind.STRUCTURAL: 2.0,
    HazardKind.CROWD_CRUSH: 0.3,
}

# Probability that word of mouth reaches an agent, scaled by local
# density and the fraction of neighbours already aware. Zero density
# (no field data) degrades this term to zero, which is why hazard
# response does not depend on the density field landing first.
RUMOR_COEFFICIENT = 0.6


@dataclass(frozen=True)
class PerceptionInputs:
    """Everything needed to decide whether one agent notices a hazard.

    Attributes:
        distance_mi: The agent's position.
        tick: Current tick.
        has_broadcast_receiver: Whether the agent can receive sirens,
            phone alerts, or emergency broadcasts.
        local_density: Agents per segment, 0..1 normalised. Optional;
            absent means word of mouth cannot operate.
        fraction_aware: Share of nearby agents already aware, 0..1.
    """

    distance_mi: float
    tick: int
    has_broadcast_receiver: bool = True
    local_density: float = 0.0
    fraction_aware: float = 0.0


def perceives(hazard: Hazard, inputs: PerceptionInputs, session_id: str) -> bool:
    """Whether this agent becomes aware of ``hazard`` this tick.

    Four independent channels; any one suffices:

    1. **Sight** -- within visual range, with probability falling off
       with distance (line of sight is imperfect).
    2. **Sound** -- within audible range, scaled by severity. Loud
       events are heard well beyond where they can be seen.
    3. **Broadcast** -- sirens/alerts reach equipped agents from
       ``warning_ticks`` before onset.
    4. **Word of mouth** -- propagates through the crowd, scaled by
       local density and how many neighbours already know.

    Deterministic: seeded from ``(session_id, hazard.id, tick)``.
    """
    if not hazard.is_active_at(inputs.tick):
        # Only a broadcast can warn before onset.
        if (
            inputs.has_broadcast_receiver
            and hazard.warning_ticks > 0
            and hazard.ticks_until_onset(inputs.tick) <= hazard.warning_ticks
        ):
            return True
        return False

    separation = abs(inputs.distance_mi - hazard.origin_mi)
    rng = random.Random(runner_seed(f"{session_id}:{hazard.id}", inputs.tick))

    # 1. Sight -- probability decays linearly to the edge of range.
    sight_range = _SIGHT_MI.get(hazard.kind, 0.5)
    if separation <= sight_range:
        if rng.random() < 1.0 - (separation / sight_range) * 0.5:
            return True

    # 2. Sound -- range scales with how violent the event is.
    if separation <= _SOUND_MI.get(hazard.kind, 1.0) * max(hazard.severity, 0.1):
        return True

    # 3. Broadcast -- alerts continue after onset.
    if inputs.has_broadcast_receiver and hazard.warning_ticks > 0:
        return True

    # 4. Word of mouth -- needs both people nearby and people who know.
    rumor_p = RUMOR_COEFFICIENT * inputs.local_density * inputs.fraction_aware
    return rng.random() < rumor_p


def blocked_destination_ids(hazards: tuple[Hazard, ...], tick: int, destinations) -> frozenset[str]:
    """Destinations made unusable by an active, travel-blocking hazard.

    Feeding this into ``goals.next_goal_after_closure`` as ``exclude_ids``
    is what makes agents reroute when their intended exit is hit.
    """
    blocked = set()
    for destination in destinations:
        if destination.distance_mi is None:
            continue
        for hazard in hazards:
            if hazard.blocks_travel and hazard.affects(destination.distance_mi, tick):
                blocked.add(destination.id)
                break
    return frozenset(blocked)


def compute_salience(
    hazard: Hazard,
    inputs: PerceptionInputs,
    option_count: int = 1,
    persona_stakes: float = 0.0,
    already_known: bool = False,
) -> float:
    """How urgently this agent needs to make a decision, 0..1.

    Blends five terms, weighted so **time-to-impact dominates**: thirty
    seconds and two hours are entirely different problems even at equal
    proximity.

    * proximity -- how close the threat is, relative to its reach;
    * urgency -- how soon it arrives (decaying in ticks remaining);
    * novelty -- new information is more salient than known information;
    * optionality -- a real choice between routes matters; a single
      option does not require deliberation;
    * stakes -- persona vulnerability (mobility-limited, accompanied by
      children, and so on).

    Used by the promotion gate: agents above the broadcast threshold get
    to reason, everyone else keeps walking deterministically.
    """
    reach = max(hazard.radius_at(inputs.tick), _SIGHT_MI.get(hazard.kind, 0.5))
    separation = abs(inputs.distance_mi - hazard.origin_mi)
    proximity = max(0.0, 1.0 - separation / reach) if reach > 0 else 0.0

    ticks_out = hazard.ticks_until_onset(inputs.tick)
    urgency = 1.0 if ticks_out == 0 else 1.0 / (1.0 + math.log1p(ticks_out))

    novelty = 0.0 if already_known else 1.0
    optionality = 0.0 if option_count <= 1 else min(1.0, (option_count - 1) / 3.0)

    score = (
        0.30 * proximity
        + 0.35 * urgency * hazard.severity
        + 0.15 * novelty
        + 0.10 * optionality
        + 0.10 * min(1.0, max(0.0, persona_stakes))
    )
    return max(0.0, min(1.0, score))


@dataclass
class PromotionBudget:
    """Admission control for how many agents may reason per tick.

    A hazard alerts everyone in its radius on the *same* tick, so naive
    per-agent reasoning produces a thundering herd against the
    simulator's fixed tick window. The simulator computes a threshold
    from the previous tick's salience distribution such that roughly
    ``budget`` agents clear it, then broadcasts that threshold. Each
    agent applies it locally -- no coordination, no extra round trips,
    one tick of lag.

    Attributes:
        budget: Target number of promoted agents per tick.
        threshold: Current salience cut-off. Starts at 1.0 so that
            nothing is promoted until a threshold has been computed.
    """

    budget: int = 8
    threshold: float = 1.0
    _history: list[float] = field(default_factory=list)

    def observe(self, saliences: list[float]) -> float:
        """Recompute the threshold from this tick's salience scores.

        Returns the threshold to broadcast for the *next* tick.
        """
        self._history = sorted(saliences, reverse=True)
        if not self._history:
            self.threshold = 1.0
        elif len(self._history) <= self.budget:
            # Everyone fits: admit any agent with non-zero salience.
            self.threshold = 0.0
        else:
            self.threshold = self._history[self.budget]
        return self.threshold

    def admits(self, salience: float) -> bool:
        """Whether an agent with this salience may reason."""
        return salience > self.threshold
