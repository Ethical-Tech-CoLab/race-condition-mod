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

"""Tests for the DUMBO pedestrian scenario.

Two things are being verified: that DUMBO expresses the behaviour the
scenario is *for* (ranked transit exits, accessibility constraints,
dwell), and that it runs on the unmodified kernel -- the proof that the
generalisation actually generalises.
"""

import random

from agents.runner.kernel import TickEnv, step
from agents.scenarios.dumbo import (
    DUMBO,
    DUMBO_PERSONAS,
    DUMBO_PHYSICS,
    FERRY_MODELLING_GAPS,
    WALKING_SPEED_SCALE,
)
from agents.scenarios.goals import next_goal_after_closure, select_goal
from agents.scenarios.spec import DestinationKind

_PERSONAS = {p.id: p for p in DUMBO_PERSONAS}


def _env(tick=1, minutes_per_tick=1.0, goal=None, session_id="dumbo-test"):
    return TickEnv(
        tick=tick,
        minutes_per_tick=minutes_per_tick,
        elapsed_minutes=tick * minutes_per_tick,
        race_distance_mi=DUMBO.profile.course_distance_mi,
        session_id=session_id,
        physics=DUMBO_PHYSICS,
        goal=goal,
    )


# --- Scenario shape ---------------------------------------------------


def test_subway_exits_outrank_the_ferry():
    """A/High St first, then F/York St, then the ferry."""
    assert [d.id for d in DUMBO.terminal_destinations()] == [
        "subway-high-st",
        "subway-york-st",
        "ferry-dumbo",
    ]


def test_default_goal_is_high_st():
    assert select_goal(DUMBO, tick=0).id == "subway-high-st"


def test_ferry_only_available_in_its_service_window():
    """The ferry is a boat: it is not always there."""
    ferry = next(d for d in DUMBO.destinations if d.id == "ferry-dumbo")
    assert not ferry.is_open_at(0)
    assert ferry.is_open_at(6)
    assert ferry.is_open_at(9)
    assert not ferry.is_open_at(10)


def test_subway_exits_are_always_open():
    for did in ("subway-high-st", "subway-york-st"):
        d = next(x for x in DUMBO.destinations if x.id == did)
        assert d.is_open_at(0) and d.is_open_at(59)


def test_all_destination_kinds_the_scenario_needs_are_present():
    kinds = {d.kind for d in DUMBO.destinations}
    assert kinds == {
        DestinationKind.EXIT_SUBWAY,
        DestinationKind.EXIT_FERRY,
        DestinationKind.TOUR_STOP,
        DestinationKind.WATER,
        DestinationKind.AID,
    }


def test_destination_ids_are_unique():
    ids = [d.id for d in DUMBO.destinations]
    assert len(ids) == len(set(ids))


def test_destinations_lie_within_the_district():
    """DUMBO is small; nothing should sit miles away."""
    for d in DUMBO.destinations:
        assert d.distance_mi is not None
        assert 0.0 <= d.distance_mi <= DUMBO.profile.course_distance_mi
        lon, lat = d.coordinates
        assert -74.01 < lon < -73.97
        assert 40.69 < lat < 40.71


# --- Accessibility ----------------------------------------------------


def test_mobility_limited_pedestrian_avoids_the_stairs_only_station():
    """High St is stairs-only, so step-free travellers route to York St."""
    goal = select_goal(DUMBO, tick=0, persona=_PERSONAS["mobility_limited"])
    assert goal.id == "subway-york-st"
    assert goal.constraints["step_free"] is True


def test_family_with_stroller_also_avoids_high_st():
    assert select_goal(DUMBO, tick=0, persona=_PERSONAS["family"]).id == "subway-york-st"


def test_commuter_takes_the_nearest_subway_and_ignores_the_ferry():
    """Even inside the ferry window, a commuter stays on the subway."""
    assert select_goal(DUMBO, tick=7, persona=_PERSONAS["commuter"]).id == "subway-high-st"


def test_tourist_prefers_the_ferry_while_it_is_running():
    """For a visitor the ferry is part of the outing, not a fallback."""
    tourist = _PERSONAS["tourist"]
    assert select_goal(DUMBO, tick=7, persona=tourist).id == "ferry-dumbo"
    # Outside the window the tourist falls back to the ranked subways.
    assert select_goal(DUMBO, tick=0, persona=tourist).id == "subway-high-st"


def test_persona_weights_are_a_distribution():
    total = sum(p.weight for p in DUMBO_PERSONAS)
    assert abs(total - 1.0) < 1e-9
    assert all(p.weight > 0 for p in DUMBO_PERSONAS)


def test_personas_have_distinct_walking_speeds():
    """Commuters outpace tourists, who outpace families and slower walkers."""
    speeds = {p.id: p.speed_mean for p in DUMBO_PERSONAS}
    assert speeds["commuter"] > speeds["tourist"] > speeds["family"] > speeds["mobility_limited"]


def test_tourists_dwell_far_more_than_commuters():
    assert _PERSONAS["tourist"].dwell_probability > 10 * _PERSONAS["commuter"].dwell_probability


# --- Exit closure -----------------------------------------------------


def test_closing_high_st_reroutes_to_york_st():
    """The scenario this pack exists to exercise."""
    high = next(d for d in DUMBO.destinations if d.id == "subway-high-st")
    rerouted = next_goal_after_closure(
        DUMBO, high, tick=0, exclude_ids=frozenset({"subway-high-st"})
    )
    assert rerouted.id == "subway-york-st"


def test_closing_both_subways_leaves_only_the_ferry_and_only_in_window():
    both_closed = frozenset({"subway-high-st", "subway-york-st"})
    assert select_goal(DUMBO, tick=7, exclude_ids=both_closed).id == "ferry-dumbo"
    # Outside the ferry window there is genuinely nowhere to go.
    assert select_goal(DUMBO, tick=0, exclude_ids=both_closed) is None


# --- Runs on the unmodified kernel ------------------------------------


def test_pedestrian_reaches_the_subway_on_the_shared_kernel():
    """The generalisation proof: same kernel, different scenario."""
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    goal = select_goal(DUMBO, tick=0)

    arrived_tick = None
    for tick in range(1, 30):
        result = step(state, _env(tick=tick, goal=goal))
        if result["runner_status"] == "finished":
            arrived_tick = tick
            assert result["arrived_at"] == "subway-high-st"
            break

    assert arrived_tick is not None, "pedestrian never reached the subway"
    # 0.30 mi at ~3.1 mph is ~6 minutes.
    assert 4 <= arrived_tick <= 8


def test_walking_pace_is_realistic():
    """velocity=1.0 should walk ~3.1 miles in an hour, not run 6.2."""
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    result = step(state, _env(tick=1, minutes_per_tick=60.0))
    assert abs(result["mi_this_tick"] - WALKING_SPEED_SCALE) < 0.01


def test_hydration_does_not_slow_pedestrians():
    """min_resource_factor=1.0 disables the marathon speed penalty."""
    dry = step({"velocity": 1.0, "distance": 0.0, "water": 0.0}, _env())
    wet = step({"velocity": 1.0, "distance": 0.0, "water": 100.0}, _env())
    assert dry["effective_velocity"] == wet["effective_velocity"]


def test_pedestrians_do_not_collapse_on_a_half_mile_walk():
    """The resource mechanic is inert here by construction."""
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    for tick in range(1, 30):
        result = step(state, _env(tick=tick))
        assert not result["collapsed"]
        assert not result["exhausted"]


def test_runs_are_reproducible_from_the_session_id():
    def walk(session_id: str) -> list:
        state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
        return [step(state, _env(tick=t, session_id=session_id)) for t in range(1, 8)]

    assert walk("ped-001") == walk("ped-001")


def test_scenario_is_independent_of_global_rng():
    def walk(seed: int) -> list:
        random.seed(seed)
        state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
        return [step(state, _env(tick=t)) for t in range(1, 6)]

    assert walk(1) == walk(424242)


# --- Documented gaps --------------------------------------------------


def test_ferry_gaps_are_documented():
    """The ferry is knowingly incomplete; keep that visible."""
    assert FERRY_MODELLING_GAPS
    assert any("capacity" in gap for gap in FERRY_MODELLING_GAPS)
    assert any("timetable" in gap or "window" in gap for gap in FERRY_MODELLING_GAPS)


def test_ferry_declares_the_constraints_it_does_model():
    ferry = next(d for d in DUMBO.destinations if d.id == "ferry-dumbo")
    assert ferry.capacity == 149
    assert ferry.open_ticks is not None
    assert ferry.constraints["step_free"] is True
