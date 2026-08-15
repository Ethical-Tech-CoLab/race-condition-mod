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

"""Tests for goal selection and arrival.

Covers the marathon special case (a single finish line, behaviour
unchanged) and the ranked/closable-exit behaviour the Dumbo and
evacuation scenarios depend on.
"""

from agents.runner.constants import MARATHON_MI
from agents.runner.kernel import TickEnv, step
from agents.scenarios.goals import (
    goal_distance_mi,
    has_arrived,
    next_goal_after_closure,
    opportunistic_destinations,
    select_goal,
)
from agents.scenarios.marathon import MARATHON, MARATHON_PHYSICS, MARATHON_PROFILE
from agents.scenarios.spec import (
    Destination,
    DestinationKind,
    PersonaSpec,
    ScenarioSpec,
    TerminationSpec,
)

# --- A Dumbo-shaped fixture: ranked subway exits + constrained ferry ---

_HIGH_ST = Destination(
    id="subway-high-st",
    kind=DestinationKind.EXIT_SUBWAY,
    name="A/C High St",
    distance_mi=0.30,
    preference_rank=0,
    capacity=200,
    constraints={"step_free": False},
)
_YORK_ST = Destination(
    id="subway-york-st",
    kind=DestinationKind.EXIT_SUBWAY,
    name="F York St",
    distance_mi=0.45,
    preference_rank=1,
    capacity=150,
    constraints={"step_free": True},
)
_FERRY = Destination(
    id="ferry-dumbo",
    kind=DestinationKind.EXIT_FERRY,
    name="DUMBO Ferry Landing",
    distance_mi=0.55,
    preference_rank=2,
    capacity=40,
    open_ticks=(10, 14),
    constraints={"step_free": True, "departs_every_ticks": 12},
)

DUMBO_LIKE = ScenarioSpec(
    id="dumbo-like",
    name="Dumbo (test fixture)",
    physics=MARATHON_PHYSICS,
    profile=MARATHON_PROFILE,
    termination=TerminationSpec(
        terminal_kinds=frozenset({DestinationKind.EXIT_SUBWAY, DestinationKind.EXIT_FERRY})
    ),
    destinations=(
        _FERRY,
        _YORK_ST,
        _HIGH_ST,
        Destination(
            id="tour-washington",
            kind=DestinationKind.TOUR_STOP,
            name="Washington St View",
            distance_mi=0.20,
        ),
    ),
)


# --- Selection ---------------------------------------------------------


def test_selects_highest_preference_exit():
    """A/High St outranks F/York St, which outranks the ferry."""
    assert select_goal(DUMBO_LIKE, tick=0).id == "subway-high-st"


def test_excluded_goal_is_skipped():
    goal = select_goal(DUMBO_LIKE, tick=0, exclude_ids=frozenset({"subway-high-st"}))
    assert goal.id == "subway-york-st"


def test_closed_destination_is_not_selected():
    """The ferry only exists inside its window."""
    only_ferry = frozenset({"subway-high-st", "subway-york-st"})
    assert select_goal(DUMBO_LIKE, tick=0, exclude_ids=only_ferry) is None
    assert select_goal(DUMBO_LIKE, tick=12, exclude_ids=only_ferry).id == "ferry-dumbo"


def test_returns_none_when_everything_is_unavailable():
    """No way out is a real evacuation state, not an error."""
    everything = frozenset({"subway-high-st", "subway-york-st", "ferry-dumbo"})
    assert select_goal(DUMBO_LIKE, tick=12, exclude_ids=everything) is None


def test_persona_constraints_filter_destinations():
    """A step-free requirement rules out the stairs-only entrance."""
    mobility = PersonaSpec(id="mobility_limited", constraints={"step_free": True})
    assert select_goal(DUMBO_LIKE, tick=0, persona=mobility).id == "subway-york-st"


def test_persona_preferred_kinds_override_scenario_rank():
    """A ferry-preferring persona takes the ferry despite its rank."""
    ferry_lover = PersonaSpec(
        id="ferry_lover",
        preferred_kinds=(DestinationKind.EXIT_FERRY, DestinationKind.EXIT_SUBWAY),
    )
    assert select_goal(DUMBO_LIKE, tick=12, persona=ferry_lover).id == "ferry-dumbo"
    # Outside the ferry window it falls back to the ranked subway exits.
    assert select_goal(DUMBO_LIKE, tick=0, persona=ferry_lover).id == "subway-high-st"


def test_selection_is_deterministic():
    """No RNG here: goal choice must not perturb seeded trajectories."""
    assert [select_goal(DUMBO_LIKE, tick=3).id for _ in range(20)] == ["subway-high-st"] * 20


def test_only_terminal_destinations_are_goals():
    """Tour stops are waypoints, never goals."""
    for _ in range(5):
        assert select_goal(DUMBO_LIKE, tick=0).kind.is_terminal


# --- Closure fall-through ---------------------------------------------


def test_goal_is_kept_while_still_open():
    assert next_goal_after_closure(DUMBO_LIKE, _HIGH_ST, tick=5).id == "subway-high-st"


def test_closure_falls_through_to_next_rank():
    """The core hazard/closure behaviour: reroute to the next choice."""
    closed_high = Destination(
        id="subway-high-st",
        kind=DestinationKind.EXIT_SUBWAY,
        distance_mi=0.30,
        preference_rank=0,
        open_ticks=(0, 4),
    )
    spec = ScenarioSpec(
        id="s",
        name="s",
        physics=MARATHON_PHYSICS,
        profile=MARATHON_PROFILE,
        termination=TerminationSpec(terminal_kinds=frozenset({DestinationKind.EXIT_SUBWAY})),
        destinations=(closed_high, _YORK_ST),
    )
    assert next_goal_after_closure(spec, closed_high, tick=2).id == "subway-high-st"
    assert next_goal_after_closure(spec, closed_high, tick=9).id == "subway-york-st"


def test_closure_returns_none_when_no_alternative():
    spec = ScenarioSpec(
        id="s",
        name="s",
        physics=MARATHON_PHYSICS,
        profile=MARATHON_PROFILE,
        termination=TerminationSpec(terminal_kinds=frozenset({DestinationKind.EXIT_SUBWAY})),
        destinations=(
            Destination(id="only", kind=DestinationKind.EXIT_SUBWAY, open_ticks=(0, 1)),
        ),
    )
    assert next_goal_after_closure(spec, spec.destinations[0], tick=9) is None


# --- Arrival ----------------------------------------------------------


def test_goal_distance_falls_back_to_course_length():
    """No goal -> the marathon comparison, unchanged."""
    assert goal_distance_mi(None, MARATHON_MI) == MARATHON_MI


def test_goal_distance_uses_goal_position():
    assert goal_distance_mi(_HIGH_ST, MARATHON_MI) == 0.30


def test_has_arrived_matches_distance_comparison():
    assert not has_arrived(0.29, _HIGH_ST, MARATHON_MI)
    assert has_arrived(0.30, _HIGH_ST, MARATHON_MI)
    assert has_arrived(0.31, _HIGH_ST, MARATHON_MI)


def test_marathon_arrival_is_the_original_comparison():
    assert not has_arrived(MARATHON_MI - 0.001, None, MARATHON_MI)
    assert has_arrived(MARATHON_MI, None, MARATHON_MI)


# --- Kernel integration -----------------------------------------------


def test_kernel_without_goal_keeps_marathon_key_set():
    """Marathon payloads must not gain goal keys."""
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    env = TickEnv(tick=1, minutes_per_tick=36.0, elapsed_minutes=36.0, race_distance_mi=MARATHON_MI)
    result = step(state, env)
    assert "goal_id" not in result
    assert "goal_kind" not in result
    assert "arrived_at" not in result


def test_kernel_finishes_early_at_a_near_goal():
    """A subway entrance at 0.3 mi ends the run far short of 26.2."""
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    env = TickEnv(
        tick=1,
        minutes_per_tick=36.0,
        elapsed_minutes=36.0,
        race_distance_mi=MARATHON_MI,
        goal=_HIGH_ST,
    )
    result = step(state, env)
    assert result["runner_status"] == "finished"
    assert result["distance"] <= 0.30
    assert result["arrived_at"] == "subway-high-st"
    assert result["goal_kind"] == DestinationKind.EXIT_SUBWAY


def test_kernel_reports_goal_before_arrival():
    """Goal is advertised every tick, arrival only once reached."""
    state = {"velocity": 0.01, "distance": 0.0, "water": 100.0}
    env = TickEnv(
        tick=1,
        minutes_per_tick=1.0,
        elapsed_minutes=1.0,
        race_distance_mi=MARATHON_MI,
        goal=_HIGH_ST,
    )
    result = step(state, env)
    assert result["goal_id"] == "subway-high-st"
    assert result["runner_status"] == "running"
    assert "arrived_at" not in result


# --- Marathon still degenerate ----------------------------------------


def test_marathon_selects_its_single_finish_line():
    assert select_goal(MARATHON, tick=0).id == "finish"


def test_marathon_water_stations_are_waypoints_not_goals():
    water = opportunistic_destinations(MARATHON, DestinationKind.WATER)
    assert water
    assert all(not d.kind.is_terminal for d in water)
    # Waypoints are ordered along the course, not by preference.
    distances = [d.distance_mi for d in water]
    assert distances == sorted(distances)


def test_marathon_goal_reproduces_course_length():
    assert goal_distance_mi(select_goal(MARATHON, tick=0), 0.0) == MARATHON_MI
