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

"""Cross-scenario eval: the same invariants applied to every scenario.

This is the regression net for the generalisation. Every scenario --
marathon, DUMBO, evacuation -- must satisfy the same structural and
runtime contracts, so adding a fourth scenario means adding one entry to
``ALL_SCENARIOS`` and inheriting the whole suite.

It complements rather than replaces the per-scenario tests:

* ``agents/tests/test_golden_marathon.py`` pins marathon's exact numbers;
* ``test_dumbo.py`` / ``test_hazard.py`` cover scenario-specific behaviour;
* this file enforces what must be true of *all* of them.

One test here (``test_persona_constraints_reference_real_destination_keys``)
exists because of a bug found during development: persona *attributes*
(group size, stakes) were mixed into persona *constraints*, which are
matched against destination properties. Every destination silently
became ineligible and goal selection returned ``None``. Cheap to guard,
easy to reintroduce.
"""

import pytest

from agents.runner.kernel import TickEnv, step
from agents.scenarios.dumbo import DUMBO
from agents.scenarios.tests.helpers import require
from agents.scenarios.goals import select_goal
from agents.scenarios.marathon import MARATHON
from agents.scenarios.mariupol import MARIUPOL
from agents.scenarios.spec import ScenarioSpec

ALL_SCENARIOS: tuple[ScenarioSpec, ...] = (MARATHON, DUMBO, MARIUPOL)
_IDS = [s.id for s in ALL_SCENARIOS]


@pytest.fixture(params=ALL_SCENARIOS, ids=_IDS)
def scenario(request) -> ScenarioSpec:
    return request.param


# --- Structural invariants --------------------------------------------


def test_scenario_ids_are_unique():
    assert len(_IDS) == len(set(_IDS))


def test_has_identity_fields(scenario):
    assert scenario.id and scenario.name and scenario.description


def test_destination_ids_are_unique(scenario):
    ids = [d.id for d in scenario.destinations]
    assert len(ids) == len(set(ids))


def test_has_at_least_one_terminal_destination(scenario):
    """Without one, agents can never finish."""
    assert scenario.terminal_destinations()


def test_terminal_destinations_match_the_declared_terminal_kinds(scenario):
    for d in scenario.terminal_destinations():
        assert d.kind in scenario.termination.terminal_kinds


def test_destinations_lie_on_the_course(scenario):
    for d in scenario.destinations:
        assert d.distance_mi is not None, f"{d.id} has no position"
        assert 0.0 <= d.distance_mi <= scenario.profile.course_distance_mi, (
            f"{d.id} at {d.distance_mi} mi is off a "
            f"{scenario.profile.course_distance_mi} mi course"
        )


def test_preference_ranks_are_non_negative(scenario):
    assert all(d.preference_rank >= 0 for d in scenario.destinations)


def test_capacities_are_positive_when_declared(scenario):
    for d in scenario.destinations:
        assert d.capacity is None or d.capacity > 0


def test_open_windows_are_well_formed(scenario):
    for d in scenario.destinations:
        if d.open_ticks is not None:
            first, last = d.open_ticks
            assert 0 <= first <= last


# --- Persona invariants -----------------------------------------------


def test_persona_weights_form_a_distribution(scenario):
    if not scenario.personas:
        pytest.skip(f"{scenario.id} uses a continuous ability scalar, not personas")
    assert abs(sum(p.weight for p in scenario.personas) - 1.0) < 1e-9
    assert all(p.weight > 0 for p in scenario.personas)


def test_persona_ids_are_unique(scenario):
    ids = [p.id for p in scenario.personas]
    assert len(ids) == len(set(ids))


def test_persona_speeds_are_positive(scenario):
    for p in scenario.personas:
        assert p.speed_mean > 0
        assert p.speed_sigma >= 0


def test_persona_dwell_probabilities_are_probabilities(scenario):
    for p in scenario.personas:
        assert 0.0 <= p.dwell_probability <= 1.0


def test_persona_constraints_reference_real_destination_keys(scenario):
    """Persona requirements must be satisfiable by some destination.

    Guards the constraints/attributes confusion described in the module
    docstring: a requirement no destination ever declares makes every
    destination ineligible and goal selection silently returns None.
    """
    declared = {key for d in scenario.destinations for key in d.constraints}
    for persona in scenario.personas:
        for key in persona.constraints:
            assert key in declared, (
                f"{scenario.id}: persona '{persona.id}' requires '{key}', which no "
                f"destination declares. Descriptive persona facts belong in "
                f"PersonaSpec.attributes, not constraints."
            )


def test_every_persona_can_reach_some_destination(scenario):
    """No persona may be stranded at the start of a run."""
    for persona in scenario.personas:
        assert select_goal(scenario, tick=0, persona=persona) is not None, (
            f"{scenario.id}: persona '{persona.id}' has no reachable destination at tick 0"
        )


# --- Physics invariants -----------------------------------------------


def test_physics_values_are_sane(scenario):
    phys = scenario.physics
    assert phys.speed_scale > 0
    assert phys.base_depletion_rate >= 0
    assert phys.fatigue_depletion_growth >= 0
    assert 0.0 <= phys.min_fatigue_factor <= 1.0
    assert 0.0 <= phys.min_resource_factor <= 1.0
    assert phys.collapse_threshold <= phys.exhaustion_threshold


def test_course_distance_is_positive(scenario):
    assert scenario.profile.course_distance_mi > 0


# --- Runtime invariants: every scenario runs on the shared kernel ------


def _run(scenario: ScenarioSpec, ticks: int = 40, session_id: str = "eval") -> list[dict]:
    """Walk one agent to its first goal on the unmodified kernel."""
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    goal = select_goal(scenario, tick=0)
    minutes = max(1.0, scenario.profile.course_distance_mi)
    results = []
    for tick in range(1, ticks + 1):
        results.append(
            step(
                state,
                TickEnv(
                    tick=tick,
                    minutes_per_tick=minutes,
                    elapsed_minutes=tick * minutes,
                    race_distance_mi=scenario.profile.course_distance_mi,
                    session_id=session_id,
                    physics=scenario.physics,
                    goal=goal,
                ),
            )
        )
        if results[-1]["runner_status"] == "finished":
            break
    return results


def test_scenario_runs_on_the_unmodified_kernel(scenario):
    """The whole point: one kernel, many scenarios, no code branches."""
    results = _run(scenario)
    assert results
    assert all(r["status"] == "success" for r in results)


def test_agent_reaches_its_goal(scenario):
    results = _run(scenario)
    assert results[-1]["runner_status"] == "finished", (
        f"{scenario.id}: agent did not reach its goal within the tick budget"
    )
    assert results[-1]["arrived_at"] == require(select_goal(scenario, tick=0)).id


def test_distance_is_monotonic(scenario):
    distances = [r["distance"] for r in _run(scenario)]
    assert distances == sorted(distances)


def test_distance_never_exceeds_the_goal(scenario):
    goal = require(select_goal(scenario, tick=0))
    goal_mi = require(goal.distance_mi, f"{goal.id} position")
    for r in _run(scenario):
        assert r["distance"] <= goal_mi + 1e-9


def test_resource_stays_in_range(scenario):
    for r in _run(scenario):
        assert 0.0 <= r["water"] <= 100.0


def test_velocity_never_negative(scenario):
    for r in _run(scenario):
        assert r["effective_velocity"] >= 0.0


def test_status_is_always_a_known_value(scenario):
    for r in _run(scenario):
        assert r["runner_status"] in {"running", "exhausted", "collapsed", "finished"}


# --- Determinism: the property every gate depends on -------------------


def test_runs_are_reproducible(scenario):
    assert _run(scenario, session_id="repeat-1") == _run(scenario, session_id="repeat-1")


def test_different_sessions_are_independent(scenario):
    """Trajectories must depend only on the session id, not on order."""
    first = _run(scenario, session_id="agent-a")
    _ = _run(scenario, session_id="agent-b")
    assert _run(scenario, session_id="agent-a") == first
