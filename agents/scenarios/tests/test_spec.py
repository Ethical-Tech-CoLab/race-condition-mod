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

"""Tests for the scenario schema and the marathon reference scenario.

The schema is the contract scenario packs are authored against, so it is
pinned here: additions are fine, silent semantic changes are not.
"""

import dataclasses

import pytest

from agents.runner import constants
from agents.runner.kernel import TickEnv, step
from agents.scenarios.marathon import (
    MARATHON,
    MARATHON_PHYSICS,
    MARATHON_PROFILE,
)
from agents.scenarios.spec import (
    Destination,
    DestinationKind,
    PersonaSpec,
    PhysicsSpec,
    ScenarioSpec,
    TerminationSpec,
)


# --- Schema shape ----------------------------------------------------


def test_all_specs_are_frozen():
    """Scenarios are configuration: they must not mutate mid-run."""
    for cls in (PhysicsSpec, Destination, PersonaSpec, TerminationSpec, ScenarioSpec):
        assert dataclasses.fields(cls) is not None
        params = getattr(cls, "__dataclass_params__")
        assert params.frozen, f"{cls.__name__} must be a frozen dataclass"


def test_destination_kinds_partition_into_terminal_and_opportunistic():
    """Every kind must be classified; none may be ambiguous."""
    terminal = {k for k in DestinationKind if k.is_terminal}
    opportunistic = {k for k in DestinationKind if not k.is_terminal}
    assert terminal and opportunistic
    assert terminal | opportunistic == set(DestinationKind)
    assert not (terminal & opportunistic)


def test_terminal_kinds_are_the_expected_set():
    """Pin the terminal set so reclassification is a conscious act."""
    assert {k for k in DestinationKind if k.is_terminal} == {
        DestinationKind.FINISH,
        DestinationKind.EXIT_SUBWAY,
        DestinationKind.EXIT_FERRY,
        DestinationKind.CORRIDOR_EXIT,
        DestinationKind.SHELTER,
    }


def test_destination_kind_serialises_as_plain_string():
    """``str`` mixin keeps tick events JSON-encodable without hooks."""
    import json

    assert json.dumps({"k": DestinationKind.EXIT_SUBWAY}) == '{"k": "exit_subway"}'


# --- Destination availability ----------------------------------------


def test_destination_open_by_default():
    assert Destination(id="d", kind=DestinationKind.FINISH).is_open_at(9999)


def test_destination_respects_open_window():
    """Closure windows are inclusive on both ends."""
    d = Destination(id="ferry", kind=DestinationKind.EXIT_FERRY, open_ticks=(10, 20))
    assert not d.is_open_at(9)
    assert d.is_open_at(10)
    assert d.is_open_at(20)
    assert not d.is_open_at(21)


def test_terminal_destinations_are_preference_ordered():
    """The ranked-exit contract: Dumbo needs A/High before F/York."""
    spec = ScenarioSpec(
        id="t",
        name="t",
        physics=MARATHON_PHYSICS,
        profile=MARATHON_PROFILE,
        termination=TerminationSpec(
            terminal_kinds=frozenset({DestinationKind.EXIT_SUBWAY, DestinationKind.EXIT_FERRY})
        ),
        destinations=(
            Destination(id="ferry", kind=DestinationKind.EXIT_FERRY, preference_rank=2),
            Destination(id="york", kind=DestinationKind.EXIT_SUBWAY, preference_rank=1),
            Destination(id="high", kind=DestinationKind.EXIT_SUBWAY, preference_rank=0),
        ),
    )
    assert [d.id for d in spec.terminal_destinations()] == ["high", "york", "ferry"]


def test_closing_preferred_exit_falls_through_to_next_rank():
    """Hazard/schedule closure is how options are removed."""
    spec = ScenarioSpec(
        id="t",
        name="t",
        physics=MARATHON_PHYSICS,
        profile=MARATHON_PROFILE,
        termination=TerminationSpec(terminal_kinds=frozenset({DestinationKind.EXIT_SUBWAY})),
        destinations=(
            Destination(id="high", kind=DestinationKind.EXIT_SUBWAY, preference_rank=0, open_ticks=(0, 5)),
            Destination(id="york", kind=DestinationKind.EXIT_SUBWAY, preference_rank=1),
        ),
    )
    assert [d.id for d in spec.open_terminal_destinations(tick=3)] == ["high", "york"]
    assert [d.id for d in spec.open_terminal_destinations(tick=9)] == ["york"]


def test_destinations_of_filters_by_kind():
    water = MARATHON.destinations_of(DestinationKind.WATER)
    assert water
    assert all(d.kind == DestinationKind.WATER for d in water)


# --- Marathon reference scenario -------------------------------------


def test_marathon_physics_matches_constants_module():
    """The spec must not drift from ``agents/runner/constants.py``.

    Values are imported, not re-typed, so this asserts the wiring rather
    than duplicating the numbers.
    """
    assert MARATHON_PHYSICS.speed_scale == constants.SPEED_SCALE
    assert MARATHON_PHYSICS.base_depletion_rate == constants.BASE_DEPLETION_RATE
    assert MARATHON_PHYSICS.fatigue_depletion_growth == constants.FATIGUE_DEPLETION_GROWTH
    assert MARATHON_PHYSICS.natural_fatigue_rate == constants.NATURAL_FATIGUE_RATE
    assert MARATHON_PHYSICS.min_fatigue_factor == constants.MIN_FATIGUE_FACTOR
    assert MARATHON_PHYSICS.exhaustion_threshold == constants.EXHAUSTION_THRESHOLD
    assert MARATHON_PHYSICS.collapse_threshold == constants.COLLAPSE_THRESHOLD
    assert MARATHON_PHYSICS.resource_station_interval_mi == constants.HYDRATION_STATION_INTERVAL_MI
    assert MARATHON_PHYSICS.resource_station_refill == constants.HYDRATION_STATION_REFILL


def test_marathon_profile_matches_constants_module():
    assert MARATHON_PROFILE.lognormal_mu == constants.LOGNORMAL_MU
    assert MARATHON_PROFILE.lognormal_sigma == constants.LOGNORMAL_SIGMA
    assert MARATHON_PROFILE.min_finish_min == constants.MIN_FINISH_MIN
    assert MARATHON_PROFILE.max_finish_min == constants.MAX_FINISH_MIN
    assert MARATHON_PROFILE.wall_hit_probability == constants.WALL_HIT_PROBABILITY
    assert MARATHON_PROFILE.course_distance_mi == constants.MARATHON_MI


def test_marathon_has_exactly_one_terminal_destination():
    """Marathon is the degenerate case: a single finish line."""
    terminal = MARATHON.terminal_destinations()
    assert len(terminal) == 1
    assert terminal[0].kind == DestinationKind.FINISH
    assert terminal[0].distance_mi == constants.MARATHON_MI


def test_marathon_water_stations_match_kernel_cadence():
    """Declared stations must line up with the kernel's marker maths.

    The kernel refills at ``int(distance / interval)`` boundaries; the
    declarative destination list has to agree or the two views of the
    course diverge.
    """
    stations = MARATHON.destinations_of(DestinationKind.WATER)
    expected = int(constants.MARATHON_MI / constants.HYDRATION_STATION_INTERVAL_MI)
    assert len(stations) == expected

    for i, station in enumerate(stations, start=1):
        assert station.distance_mi == pytest.approx(i * constants.HYDRATION_STATION_INTERVAL_MI, abs=1e-4)


def test_marathon_stations_lie_on_the_course():
    for d in MARATHON.destinations:
        assert d.distance_mi is not None
        assert 0.0 <= d.distance_mi <= constants.MARATHON_MI


def test_marathon_has_no_personas():
    """Ability is a continuous scalar here, not a categorical archetype."""
    assert MARATHON.personas == ()


def test_marathon_destination_ids_are_unique():
    ids = [d.id for d in MARATHON.destinations]
    assert len(ids) == len(set(ids))


# --- Kernel integration ----------------------------------------------


def test_kernel_defaults_to_marathon_physics():
    """Existing callers get marathon dynamics without passing a spec."""
    env = TickEnv(tick=1, minutes_per_tick=36.0, elapsed_minutes=36.0, race_distance_mi=constants.MARATHON_MI)
    assert env.physics is MARATHON_PHYSICS


def test_kernel_honours_an_alternative_physics_spec():
    """The generalisation is real: swapping the spec changes dynamics.

    ``mi_this_tick`` is rounded to 3 decimals in the payload, so the
    tolerance is set to that rounding granularity rather than to float
    precision.
    """
    slow = dataclasses.replace(MARATHON_PHYSICS, speed_scale=constants.SPEED_SCALE / 2)

    def run(physics):
        state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
        env = TickEnv(
            tick=1,
            minutes_per_tick=60.0,
            elapsed_minutes=60.0,
            race_distance_mi=constants.MARATHON_MI,
            physics=physics,
        )
        return step(state, env)["mi_this_tick"]

    assert run(slow) == pytest.approx(run(MARATHON_PHYSICS) / 2, abs=1e-3)


def test_min_resource_factor_generalises_the_dehydration_penalty():
    """``min_resource_factor=1.0`` disables the penalty entirely."""
    no_penalty = dataclasses.replace(MARATHON_PHYSICS, min_resource_factor=1.0)
    env = TickEnv(
        tick=1,
        minutes_per_tick=36.0,
        elapsed_minutes=36.0,
        race_distance_mi=constants.MARATHON_MI,
        physics=no_penalty,
    )
    dry = step({"velocity": 1.0, "distance": 0.0, "water": 0.0}, env)
    wet = step({"velocity": 1.0, "distance": 0.0, "water": 100.0}, env)
    assert dry["effective_velocity"] == wet["effective_velocity"]
