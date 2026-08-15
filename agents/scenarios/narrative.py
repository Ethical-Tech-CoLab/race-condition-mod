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

"""Persona-pooled narrative: agent inner monologue without inference.

Replaces per-agent LLM narration with a **static, reviewed corpus**
indexed by the seeded RNG the simulation already uses.

Why pooling is sound here
-------------------------
``inner_thought`` is **non-causal**: it is a display string that never
affects dynamics. It is also produced far faster than it can be
consumed. At 1000 agents over 12 ticks the simulation generates ~100
thoughts per second, while the UI shows the camera-followed agent's
thought and rotates it every 5 seconds -- roughly 0.2 per second. That
is ~500x overproduction of an output that cannot change the simulation.

The repository already concedes this: the frontend ships
``EXTERNAL_THOUGHTS``, 65 hand-written strings cycled with shuffle-bag
semantics, and falls back to them whenever an agent-authored thought is
absent. This module makes that the primary mechanism rather than the
fallback, and makes it persona- and situation-aware.

Why it is *required* for the evacuation scenario
------------------------------------------------
For a scenario shaped by real events, a static corpus is not a cost
compromise -- it is the correct design:

* it can be reviewed by subject-matter experts and affected communities
  **before** anyone sees it;
* it is deterministic, so any run is reproducible and citable;
* it removes any possibility of a model improvising about a real siege.

Live inference stays reserved for the small number of promoted agents
making genuinely causal routing decisions (see
``agents/scenarios/hazard.py``).

Determinism
-----------
Selection is seeded from ``(session_id, persona, phase, tick)``, so the
same run always produces the same narrative. No global RNG is touched.
"""

import random
from dataclasses import dataclass, field
from enum import Enum

from agents.runner.constants import runner_seed


class NarrativePhase(str, Enum):
    """Coarse progress bands. Thoughts differ by where you are, not by mile."""

    START = "start"
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"
    ARRIVED = "arrived"


class ResourceBucket(str, Enum):
    """Coarse resource bands (stamina/hydration)."""

    FRESH = "fresh"
    OK = "ok"
    LOW = "low"
    CRITICAL = "critical"


def derive_phase(distance_mi: float, goal_distance_mi: float, arrived: bool = False) -> NarrativePhase:
    """Bucket progress toward the goal into a narrative phase."""
    if arrived:
        return NarrativePhase.ARRIVED
    if goal_distance_mi <= 0:
        return NarrativePhase.START
    fraction = distance_mi / goal_distance_mi
    if fraction <= 0.02:
        return NarrativePhase.START
    if fraction < 0.35:
        return NarrativePhase.EARLY
    if fraction < 0.75:
        return NarrativePhase.MIDDLE
    return NarrativePhase.LATE


def derive_bucket(resource: float) -> ResourceBucket:
    """Bucket a 0-100 resource level."""
    if resource >= 75.0:
        return ResourceBucket.FRESH
    if resource >= 45.0:
        return ResourceBucket.OK
    if resource >= 20.0:
        return ResourceBucket.LOW
    return ResourceBucket.CRITICAL


# Pool keys are ``(persona_id | None, phase | None)``. ``None`` acts as a
# wildcard, giving a fallback chain from most to least specific. This
# keeps authoring tractable: write specific lines only where a persona's
# voice genuinely differs, and let everything else inherit.
PoolKey = tuple[str | None, NarrativePhase | None]


@dataclass(frozen=True)
class NarrativePool:
    """A reviewed corpus of inner-monologue lines for one scenario.

    Attributes:
        scenario_id: Scenario this corpus belongs to.
        entries: Mapping of ``(persona, phase)`` to candidate lines.
        reviewed: Whether the corpus has been signed off. Scenarios
            depicting real events should refuse to render narrative
            until this is ``True``.
    """

    scenario_id: str
    entries: dict[PoolKey, tuple[str, ...]] = field(default_factory=dict)
    reviewed: bool = False

    def candidates(self, persona: str | None, phase: NarrativePhase) -> tuple[str, ...]:
        """Lines for this situation, most specific match first.

        Falls back ``(persona, phase)`` -> ``(persona, any)`` ->
        ``(any, phase)`` -> ``(any, any)``.
        """
        for key in ((persona, phase), (persona, None), (None, phase), (None, None)):
            found = self.entries.get(key)
            if found:
                return found
        return ()

    def all_lines(self) -> tuple[str, ...]:
        """Every line in the corpus, for review and coverage checks."""
        return tuple(line for lines in self.entries.values() for line in lines)


def thought_for(
    pool: NarrativePool,
    session_id: str,
    tick: int,
    persona: str | None = None,
    phase: NarrativePhase = NarrativePhase.EARLY,
    bucket: ResourceBucket = ResourceBucket.FRESH,
) -> str:
    """Pick a line deterministically for this agent, tick, and situation.

    Returns ``""`` when the corpus is unreviewed or has no candidates --
    callers must treat an empty thought as valid (the UI already does).

    ``bucket`` participates in the seed rather than the lookup, so an
    agent's thought changes as its condition changes without requiring
    a separately authored corpus per resource band.
    """
    if not pool.reviewed:
        return ""
    lines = pool.candidates(persona, phase)
    if not lines:
        return ""
    seed_key = f"{session_id}:{persona or '-'}:{phase.value}:{bucket.value}"
    rng = random.Random(runner_seed(seed_key, tick))
    return lines[rng.randrange(len(lines))]


def thought_for_state(pool: NarrativePool, state, session_id: str, tick: int, goal_distance_mi: float) -> str:
    """Convenience wrapper: derive phase and bucket from agent state."""
    return thought_for(
        pool,
        session_id=session_id,
        tick=tick,
        persona=state.get("persona"),
        phase=derive_phase(
            state.get("distance", 0.0),
            goal_distance_mi,
            arrived=bool(state.get("finished")),
        ),
        bucket=derive_bucket(state.get("water", 100.0)),
    )


# --------------------------------------------------------------------
# Marathon corpus
# --------------------------------------------------------------------
# Voice: competitive, wry, physical. Mirrors the tone of the frontend's
# existing EXTERNAL_THOUGHTS pool.
MARATHON_POOL = NarrativePool(
    scenario_id="marathon",
    reviewed=True,
    entries={
        (None, NarrativePhase.START): (
            "Corral nerves, deep breath.",
            "Watch started. Here we go.",
            "Don't go out too fast.",
            "Legs feel springy today.",
            "Everyone looks faster than me.",
        ),
        (None, NarrativePhase.EARLY): (
            "Settle into the pace.",
            "Crowd noise is medicine.",
            "This feels sustainable.",
            "Save it for later.",
            "Found my rhythm early.",
        ),
        (None, NarrativePhase.MIDDLE): (
            "Halfway. Still honest.",
            "Legs heavier than mile six.",
            "Bargaining with my quads.",
            "Just get to the next mile.",
            "Someone said I look strong.",
        ),
        (None, NarrativePhase.LATE): (
            "Lungs burning, push through.",
            "Everything hurts. Keep moving.",
            "I can hear the finish.",
            "Three miles is nothing.",
            "Do not walk. Do not walk.",
        ),
        (None, NarrativePhase.ARRIVED): (
            "Done. Never again. Probably.",
            "Where is the water?",
            "Legs, you may stop now.",
            "That medal weighs nothing.",
        ),
    },
)

# --------------------------------------------------------------------
# DUMBO corpus
# --------------------------------------------------------------------
# Voice: everyday, distracted, place-specific. Commuters and visitors
# genuinely differ here, so both get their own lines.
DUMBO_POOL = NarrativePool(
    scenario_id="dumbo",
    reviewed=True,
    entries={
        ("commuter", None): (
            "High Street, six minutes.",
            "Tourists on the whole sidewalk.",
            "If I miss this train...",
            "Same walk, every day.",
            "Left at the cobblestones.",
        ),
        ("tourist", None): (
            "The bridge looks unreal.",
            "One more photo. Just one.",
            "Is this the carousel street?",
            "Everything here is cobblestone.",
            "Ferry or subway? Ferry.",
        ),
        ("family", None): (
            "Hold my hand, please.",
            "Stroller versus cobblestones again.",
            "Anyone need the bathroom?",
            "Elevator, not the stairs.",
            "Five more minutes, then home.",
        ),
        ("mobility_limited", None): (
            "York Street has the elevator.",
            "These cobbles are rough.",
            "Taking my time today.",
            "Looking for the ramp.",
        ),
        (None, NarrativePhase.ARRIVED): (
            "Made it. Finally sitting.",
            "Train's coming. Good timing.",
            "Out of the wind at last.",
        ),
        (None, None): (
            "River wind is cold.",
            "Everyone's taking the same photo.",
            "Which way to the station?",
        ),
    },
)

# --------------------------------------------------------------------
# Evacuation corpus
# --------------------------------------------------------------------
# Voice: practical, restrained, oriented to logistics and family rather
# than to danger. Deliberately avoids depicting violence, injury, or
# distress. These are people solving immediate problems -- where to go,
# who to keep track of, what to carry.
#
# ``reviewed=False`` on purpose: this scenario is shaped by real events,
# and no narrative should render until a human with the relevant
# expertise has signed the corpus off. ``thought_for`` returns "" until
# then, so the simulation runs correctly with no narrative at all.
MARIUPOL_POOL = NarrativePool(
    scenario_id="mariupol",
    reviewed=False,
    entries={
        (None, NarrativePhase.START): (
            "Documents, water, one bag.",
            "Everyone accounted for?",
            "Which road is open today?",
            "Keep the group together.",
        ),
        (None, NarrativePhase.EARLY): (
            "Follow the people ahead.",
            "Checking the group again.",
            "Slower than I hoped.",
            "Stay on the main road.",
        ),
        (None, NarrativePhase.MIDDLE): (
            "Rest a moment, then continue.",
            "Still together. Keep going.",
            "The bag is heavier now.",
            "Someone said the corridor's open.",
        ),
        (None, NarrativePhase.LATE): (
            "Almost at the assembly point.",
            "Nearly there. Keep walking.",
            "Lights ahead. That's it.",
        ),
        (None, NarrativePhase.ARRIVED): (
            "We're here. All of us.",
            "Sitting down for a moment.",
            "Somewhere to wait, finally.",
        ),
        ("family_with_children", None): (
            "Hold on to me.",
            "Count them again. Four.",
            "Nearly there, I promise.",
        ),
        ("elderly", None): (
            "One step, then another.",
            "Need to rest soon.",
            "I know this street.",
        ),
        ("mobility_limited", None): (
            "Is there a smoother route?",
            "Waiting for the transport.",
            "Slow, but still moving.",
        ),
    },
)

ALL_POOLS: dict[str, NarrativePool] = {
    "marathon": MARATHON_POOL,
    "dumbo": DUMBO_POOL,
    "mariupol": MARIUPOL_POOL,
}


def pool_for(scenario_id: str) -> NarrativePool:
    """Look up a scenario's corpus, or an empty unreviewed one."""
    return ALL_POOLS.get(scenario_id, NarrativePool(scenario_id=scenario_id))
