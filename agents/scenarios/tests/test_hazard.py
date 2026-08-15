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

"""Tests for hazards, perception, salience, and the evacuation scenario.

Includes explicit guards on the ethical boundary: hazards must affect
routing and attention only, and the evacuation scenario must never be
able to mark an agent "collapsed".
"""

import random

from agents.runner.kernel import TickEnv, step
from agents.scenarios.tests.helpers import require
from agents.scenarios.goals import next_goal_after_closure, select_goal
from agents.scenarios.hazard import (
    Hazard,
    HazardKind,
    PerceptionInputs,
    PromotionBudget,
    blocked_destination_ids,
    compute_salience,
    perceives,
)
from agents.scenarios.mariupol import (
    MARIUPOL,
    MARIUPOL_HAZARDS,
    MARIUPOL_PERSONAS,
    MARIUPOL_PHYSICS,
    MODELLING_LIMITS,
)
from agents.scenarios.spec import DestinationKind

_PERSONAS = {p.id: p for p in MARIUPOL_PERSONAS}

_SHELL = Hazard(
    id="shell",
    kind=HazardKind.SHELLING,
    onset_tick=5,
    origin_mi=2.0,
    severity=0.9,
    initial_radius_mi=0.5,
    warning_ticks=1,
)
_ADVANCE = Hazard(
    id="advance",
    kind=HazardKind.TROOP_MOVEMENT,
    onset_tick=10,
    origin_mi=5.0,
    severity=0.7,
    initial_radius_mi=0.5,
    growth_mi_per_tick=0.25,
    warning_ticks=6,
)


# --- Hazard geometry ---------------------------------------------------


def test_hazard_inactive_before_onset():
    assert not _SHELL.is_active_at(4)
    assert _SHELL.is_active_at(5)
    assert _SHELL.radius_at(4) == 0.0


def test_static_hazard_radius_does_not_grow():
    assert _SHELL.radius_at(5) == _SHELL.radius_at(50) == 0.5


def test_advancing_hazard_radius_grows():
    assert _ADVANCE.radius_at(10) == 0.5
    assert _ADVANCE.radius_at(14) == 0.5 + 4 * 0.25


def test_affects_only_within_radius():
    assert _SHELL.affects(2.0, 5)
    assert _SHELL.affects(2.5, 5)
    assert not _SHELL.affects(2.6, 5)
    assert not _SHELL.affects(2.0, 4)  # before onset


def test_intensity_decays_from_the_origin():
    centre = _SHELL.intensity_at(2.0, 5)
    edge = _SHELL.intensity_at(2.4, 5)
    assert centre > edge > 0.0
    assert _SHELL.intensity_at(3.0, 5) == 0.0


def test_ticks_until_onset():
    assert _SHELL.ticks_until_onset(0) == 5
    assert _SHELL.ticks_until_onset(5) == 0
    assert _SHELL.ticks_until_onset(9) == 0


# --- Perception --------------------------------------------------------


def test_broadcast_warns_before_onset():
    """Sirens are the only channel that can warn ahead of the event."""
    inputs = PerceptionInputs(distance_mi=1.9, tick=4, has_broadcast_receiver=True)
    assert perceives(_SHELL, inputs, "agent-1")


def test_no_warning_without_a_receiver_before_onset():
    inputs = PerceptionInputs(distance_mi=1.9, tick=4, has_broadcast_receiver=False)
    assert not perceives(_SHELL, inputs, "agent-1")


def test_long_warning_hazard_is_known_well_in_advance():
    inputs = PerceptionInputs(distance_mi=0.0, tick=5, has_broadcast_receiver=True)
    assert perceives(_ADVANCE, inputs, "agent-1")


def test_loud_hazard_is_heard_beyond_sight():
    """Shelling carries ~4 miles: far outside its blast radius."""
    inputs = PerceptionInputs(distance_mi=5.0, tick=6, has_broadcast_receiver=False)
    assert perceives(_SHELL, inputs, "agent-1")


def test_distant_agent_without_a_receiver_may_miss_a_quiet_hazard():
    quiet = Hazard(
        id="quiet",
        kind=HazardKind.FLOOD,
        onset_tick=1,
        origin_mi=0.0,
        severity=0.2,
        warning_ticks=0,
    )
    inputs = PerceptionInputs(distance_mi=6.0, tick=2, has_broadcast_receiver=False)
    assert not perceives(quiet, inputs, "agent-1")


def test_word_of_mouth_requires_both_density_and_awareness():
    """Degrades to zero without density -- so hazards work without it."""
    quiet = Hazard(id="q", kind=HazardKind.FLOOD, onset_tick=1, origin_mi=0.0, severity=0.2)
    far = {"distance_mi": 6.0, "tick": 2, "has_broadcast_receiver": False}

    alone = PerceptionInputs(**far, local_density=0.0, fraction_aware=1.0)
    assert not perceives(quiet, alone, "agent-x")

    crowded = PerceptionInputs(**far, local_density=1.0, fraction_aware=1.0)
    aware = [perceives(quiet, crowded, f"agent-{i}") for i in range(60)]
    assert any(aware), "word of mouth should reach someone in a dense, aware crowd"


def test_perception_is_deterministic():
    inputs = PerceptionInputs(distance_mi=2.4, tick=6, has_broadcast_receiver=False)
    first = [perceives(_SHELL, inputs, f"a-{i}") for i in range(40)]
    second = [perceives(_SHELL, inputs, f"a-{i}") for i in range(40)]
    assert first == second


def test_perception_ignores_the_global_rng():
    inputs = PerceptionInputs(distance_mi=2.4, tick=6, has_broadcast_receiver=False)
    random.seed(1)
    first = [perceives(_SHELL, inputs, f"a-{i}") for i in range(40)]
    random.seed(987654)
    second = [perceives(_SHELL, inputs, f"a-{i}") for i in range(40)]
    assert first == second


# --- Closure and rerouting --------------------------------------------


def test_hazard_blocks_destinations_inside_it():
    blocked = blocked_destination_ids((_SHELL,), tick=5, destinations=MARIUPOL.destinations)
    # The central aid point sits at 2.0 mi, the hazard origin.
    assert "aid-central" in blocked
    # The western corridor at 7.5 mi is far outside it.
    assert "corridor-west" not in blocked


def test_advancing_hazard_eventually_blocks_the_northern_corridor():
    """The northern assembly point sits at 5.8 mi; the advance reaches it."""
    early = blocked_destination_ids((_ADVANCE,), tick=10, destinations=MARIUPOL.destinations)
    assert "corridor-north" not in early

    later = blocked_destination_ids((_ADVANCE,), tick=16, destinations=MARIUPOL.destinations)
    assert "corridor-north" in later


def test_agents_reroute_when_their_corridor_is_blocked():
    """The behaviour the scenario exists to exercise."""
    north = next(d for d in MARIUPOL.destinations if d.id == "corridor-north")
    blocked = blocked_destination_ids((_ADVANCE,), tick=16, destinations=MARIUPOL.destinations)
    rerouted = next_goal_after_closure(MARIUPOL, north, tick=16, exclude_ids=blocked)
    assert rerouted is not None
    assert rerouted.id != "corridor-north"


def test_non_blocking_hazard_closes_nothing():
    scare = Hazard(
        id="scare",
        kind=HazardKind.SHELLING,
        onset_tick=1,
        origin_mi=2.0,
        initial_radius_mi=2.0,
        blocks_travel=False,
    )
    assert blocked_destination_ids((scare,), tick=5, destinations=MARIUPOL.destinations) == frozenset()


# --- Salience ----------------------------------------------------------


def test_imminent_hazard_is_more_salient_than_a_distant_one():
    near = PerceptionInputs(distance_mi=2.0, tick=5)
    far = PerceptionInputs(distance_mi=2.0, tick=0)
    assert compute_salience(_SHELL, near) > compute_salience(_SHELL, far)


def test_proximity_increases_salience():
    close = PerceptionInputs(distance_mi=2.0, tick=5)
    edge = PerceptionInputs(distance_mi=3.5, tick=5)
    assert compute_salience(_SHELL, close) > compute_salience(_SHELL, edge)


def test_known_hazards_are_less_salient_than_new_ones():
    inputs = PerceptionInputs(distance_mi=2.0, tick=5)
    assert compute_salience(_SHELL, inputs, already_known=True) < compute_salience(_SHELL, inputs)


def test_having_options_raises_salience():
    """No choice means nothing to deliberate about."""
    inputs = PerceptionInputs(distance_mi=2.0, tick=5)
    assert compute_salience(_SHELL, inputs, option_count=4) > compute_salience(_SHELL, inputs, option_count=1)


def test_vulnerable_personas_are_more_salient():
    inputs = PerceptionInputs(distance_mi=2.0, tick=5)
    assert compute_salience(_SHELL, inputs, persona_stakes=1.0) > compute_salience(
        _SHELL, inputs, persona_stakes=0.0
    )


def test_salience_stays_in_range():
    for tick in range(0, 20):
        for pos in (0.0, 2.0, 4.0, 7.5):
            s = compute_salience(
                _SHELL, PerceptionInputs(distance_mi=pos, tick=tick), option_count=5, persona_stakes=1.0
            )
            assert 0.0 <= s <= 1.0


# --- Promotion budget --------------------------------------------------


def test_budget_admits_about_the_target_number():
    """The thundering-herd control: a synchronised alert is capped."""
    budget = PromotionBudget(budget=5)
    saliences = [i / 100.0 for i in range(100)]  # 100 agents alerted at once
    budget.observe(saliences)
    admitted = [s for s in saliences if budget.admits(s)]
    assert len(admitted) == 5


def test_budget_admits_everyone_when_under_capacity():
    budget = PromotionBudget(budget=10)
    saliences = [0.2, 0.5, 0.9]
    budget.observe(saliences)
    assert all(budget.admits(s) for s in saliences if s > 0)


def test_nothing_is_promoted_before_a_threshold_exists():
    """Fail closed: no reasoning until the simulator has broadcast a cut."""
    assert not PromotionBudget().admits(0.99)


def test_budget_promotes_the_most_salient_agents():
    budget = PromotionBudget(budget=3)
    saliences = [0.1, 0.95, 0.3, 0.99, 0.05, 0.97]
    budget.observe(saliences)
    admitted = sorted((s for s in saliences if budget.admits(s)), reverse=True)
    assert admitted == [0.99, 0.97, 0.95]


# --- Evacuation scenario shape ----------------------------------------


def test_corridors_outrank_shelters():
    assert [d.id for d in MARIUPOL.terminal_destinations()] == [
        "corridor-west",
        "corridor-north",
        "shelter-theatre",
        "shelter-school",
    ]


def test_corridors_close_and_shelters_remain():
    """Corridors are withdrawn; sheltering stays available."""
    assert require(select_goal(MARIUPOL, tick=10)).id == "corridor-west"
    assert require(select_goal(MARIUPOL, tick=50)).id == "shelter-theatre"


def test_mobility_limited_evacuees_prefer_shelter():
    goal = require(select_goal(MARIUPOL, tick=5, persona=_PERSONAS["mobility_limited"]))
    assert goal.kind == DestinationKind.SHELTER


def test_elderly_prefer_shelter_over_a_long_walk():
    assert require(select_goal(MARIUPOL, tick=5, persona=_PERSONAS["elderly"])).kind == DestinationKind.SHELTER


def test_step_free_requirement_excludes_the_school_basement():
    """Only the step-free shelter is eligible for wheelchair users."""
    goal = select_goal(
        MARIUPOL,
        tick=50,
        persona=_PERSONAS["mobility_limited"],
        exclude_ids=frozenset({"shelter-theatre"}),
    )
    assert goal is None


def test_persona_weights_are_a_distribution():
    assert abs(sum(p.weight for p in MARIUPOL_PERSONAS) - 1.0) < 1e-9


def test_scenario_hazards_cover_distinct_archetypes():
    kinds = {h.kind for h in MARIUPOL_HAZARDS}
    assert kinds == {HazardKind.SHELLING, HazardKind.TROOP_MOVEMENT, HazardKind.FIRE}


def test_sudden_and_slow_hazards_have_different_warning_times():
    shelling = next(h for h in MARIUPOL_HAZARDS if h.kind == HazardKind.SHELLING)
    advance = next(h for h in MARIUPOL_HAZARDS if h.kind == HazardKind.TROOP_MOVEMENT)
    assert shelling.warning_ticks < advance.warning_ticks


# --- Ethical boundary (enforced, not merely documented) ---------------


def test_evacuation_can_never_mark_an_agent_collapsed():
    """Attrition is out of scope: the threshold makes it unreachable."""
    assert MARIUPOL_PHYSICS.collapse_threshold == 0.0

    state = {"velocity": 1.0, "distance": 0.0, "water": 0.0}
    for tick in range(1, 40):
        result = step(
            state,
            TickEnv(
                tick=tick,
                minutes_per_tick=5.0,
                elapsed_minutes=tick * 5.0,
                race_distance_mi=MARIUPOL.profile.course_distance_mi,
                session_id="evac-1",
                physics=MARIUPOL_PHYSICS,
            ),
        )
        assert not result["collapsed"], "evacuation must never model casualties"


def test_hazards_do_not_appear_in_kernel_dynamics():
    """Hazards affect routing/attention, never the physics payload."""
    state = {"velocity": 1.0, "distance": 2.0, "water": 100.0}
    result = step(
        state,
        TickEnv(
            tick=6,
            minutes_per_tick=5.0,
            elapsed_minutes=30.0,
            race_distance_mi=MARIUPOL.profile.course_distance_mi,
            physics=MARIUPOL_PHYSICS,
        ),
    )
    for key in result:
        assert "hazard" not in key
        assert "casualt" not in key


def test_modelling_limits_are_declared():
    assert len(MODELLING_LIMITS) > 0
    assert any("attrition" in limit or "casualt" in limit for limit in MODELLING_LIMITS)


# --- Runs on the shared kernel ----------------------------------------


def test_evacuee_reaches_a_corridor_on_the_shared_kernel():
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    goal = select_goal(MARIUPOL, tick=0)

    arrived = False
    for tick in range(1, 100):
        result = step(
            state,
            TickEnv(
                tick=tick,
                minutes_per_tick=5.0,
                elapsed_minutes=tick * 5.0,
                race_distance_mi=MARIUPOL.profile.course_distance_mi,
                session_id="evac-2",
                physics=MARIUPOL_PHYSICS,
                goal=goal,
            ),
        )
        if result["runner_status"] == "finished":
            assert result["arrived_at"] == "corridor-west"
            arrived = True
            break
    assert arrived, "evacuee never reached the corridor"
