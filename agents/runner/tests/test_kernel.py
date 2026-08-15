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

"""Unit tests for the pure physics kernel.

These use a plain ``dict`` for state and no ``ToolContext``, no Redis,
no async -- demonstrating (and locking in) the property the extraction
was performed to obtain: **the dynamics are testable without the agent
runtime**.
"""

import random

from agents.runner.constants import (
    COLLAPSE_THRESHOLD,
    EXHAUSTION_THRESHOLD,
    MARATHON_MI,
    SPEED_SCALE,
    runner_seed,
)
from agents.runner.initialization import initialize_runner
from agents.runner.kernel import WAVE_STAGGER_SECONDS, TickEnv, step


def _env(tick=1, minutes_per_tick=36.0, race_distance_mi=MARATHON_MI, session_id="kernel-test"):
    return TickEnv(
        tick=tick,
        minutes_per_tick=minutes_per_tick,
        elapsed_minutes=tick * minutes_per_tick,
        race_distance_mi=race_distance_mi,
        session_id=session_id,
    )


def test_step_runs_without_adk_or_redis():
    """The headline property: physics needs only a dict."""
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    result = step(state, _env())
    assert result["status"] == "success"
    assert result["distance"] > 0.0


def test_step_mutates_state_in_place():
    """Matches the previous ``tool_context.state`` mutation contract."""
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    step(state, _env())
    assert state["distance"] > 0.0
    assert state["water"] < 100.0
    assert state["runner_status"] == "running"


def test_distance_matches_velocity_formula():
    """distance = v * SPEED_SCALE / 60 * minutes, with full water/no wall.

    At tick 1 with water=100 the only degradation is natural fatigue
    (0.2%/tick), so the closed form is checkable by hand.
    """
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    env = _env(tick=1, minutes_per_tick=60.0)
    result = step(state, env)

    hydration_factor = 1.0  # water == 100
    fatigue_factor = 1.0 - 0.002 * 1
    expected_mi = 1.0 * hydration_factor * fatigue_factor * SPEED_SCALE / 60.0 * 60.0
    assert result["mi_this_tick"] == round(expected_mi, 3)


def test_dehydration_halves_speed_at_zero_water():
    """hydration_factor spans 1.0 (full) to 0.5 (empty)."""
    wet = {"velocity": 1.0, "distance": 0.0, "water": 100.0}
    dry = {"velocity": 1.0, "distance": 0.0, "water": 0.0}
    wet_result = step(wet, _env())
    dry_result = step(dry, _env())
    assert dry_result["effective_velocity"] == wet_result["effective_velocity"] * 0.5


def test_wall_applies_only_past_wall_mile():
    """Wall degradation activates strictly after ``wall_mi``."""
    before = {
        "velocity": 1.0, "distance": 10.0, "water": 100.0,
        "will_hit_wall": True, "wall_mi": 18.0, "wall_severity": 0.5,
    }
    after = dict(before, distance=19.0)
    assert step(after, _env())["effective_velocity"] < step(before, _env())["effective_velocity"]


def test_wall_ignored_when_will_hit_wall_false():
    state = {
        "velocity": 1.0, "distance": 19.0, "water": 100.0,
        "will_hit_wall": False, "wall_mi": 18.0, "wall_severity": 0.5,
    }
    control = {"velocity": 1.0, "distance": 19.0, "water": 100.0}
    assert step(state, _env())["effective_velocity"] == step(control, _env())["effective_velocity"]


def test_finish_clamps_distance_and_sets_status():
    """Distance never exceeds the course; finish time is interpolated."""
    state = {"velocity": 5.0, "distance": MARATHON_MI - 0.1, "water": 100.0}
    result = step(state, _env(tick=5))
    assert result["runner_status"] == "finished"
    assert result["distance"] <= MARATHON_MI
    assert state["finished"] is True
    assert result["finish_time_minutes"] is not None
    assert result["pace_min_per_mi"] is not None


def test_finished_runner_is_noop_but_still_reports():
    """Finished runners keep reporting so aggregation still sees them."""
    state = {"velocity": 1.0, "distance": MARATHON_MI, "water": 50.0, "finished": True}
    result = step(state, _env(tick=9))
    assert result["runner_status"] == "finished"
    assert result["mi_this_tick"] == 0.0
    assert result["effective_velocity"] == 0.0


def test_collapsed_runner_is_noop():
    state = {"velocity": 1.0, "distance": 5.0, "water": 2.0, "collapsed": True}
    result = step(state, _env(tick=4))
    assert result["runner_status"] == "collapsed"
    assert result["mi_this_tick"] == 0.0


def test_exhaustion_and_collapse_thresholds():
    """< 30% water = exhausted; < 10% while exhausted = collapsed."""
    # No hydration station is crossed (distance barely moves) so the
    # thresholds are exercised directly.
    exhausted = {"velocity": 0.001, "distance": 0.0, "water": EXHAUSTION_THRESHOLD - 1}
    assert step(exhausted, _env())["exhausted"] is True

    collapsed = {"velocity": 0.001, "distance": 0.0, "water": COLLAPSE_THRESHOLD - 1}
    result = step(collapsed, _env())
    assert result["exhausted"] is True
    assert result["collapsed"] is True
    assert result["runner_status"] == "collapsed"


def test_hydration_is_seeded_not_global_random():
    """Station drinking must not consume the global RNG.

    Guards the determinism property the golden fixture depends on: if
    the kernel used ``random.random()``, seeding the global RNG
    differently would change the trajectory.
    """
    def run_with_global_seed(seed: int) -> dict:
        random.seed(seed)
        state = {"velocity": 1.0, "distance": 0.0, "water": 70.0, "hydration_efficiency": 1.0}
        return step(state, _env(tick=1, minutes_per_tick=60.0))

    assert run_with_global_seed(1) == run_with_global_seed(999999)


def test_same_session_id_gives_same_trajectory():
    """Determinism is a function of session_id, not call order."""
    def trajectory(session_id: str) -> list:
        state: dict = {}
        initialize_runner(state, session_id, 8)
        return [step(state, _env(tick=t, session_id=session_id)) for t in range(1, 6)]

    assert trajectory("repeatable-abc") == trajectory("repeatable-abc")


def test_different_session_ids_diverge():
    """Sanity: the seed actually matters."""
    def profile(session_id: str) -> float:
        state: dict = {}
        initialize_runner(state, session_id, 8)
        return state["target_finish_minutes"]

    assert profile("runner-aaa") != profile("runner-zzz")


def test_wave_stagger_only_on_first_movement_tick():
    """``gateway_delay_seconds`` appears only at tick 1 for waves > 0."""
    state = {"velocity": 1.0, "distance": 0.0, "water": 100.0, "wave_number": 3}
    first = step(dict(state), _env(tick=1))
    assert first["gateway_delay_seconds"] == 3 * WAVE_STAGGER_SECONDS

    later = step(dict(state), _env(tick=2))
    assert "gateway_delay_seconds" not in later

    wave_zero = step({"velocity": 1.0, "distance": 0.0, "water": 100.0, "wave_number": 0}, _env(tick=1))
    assert "gateway_delay_seconds" not in wave_zero


def test_tick_env_is_immutable():
    """Frozen dataclass: a tick's inputs cannot change mid-step."""
    import dataclasses

    env = _env()
    try:
        env.tick = 99  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("TickEnv must be frozen")


def test_session_id_default_matches_previous_fallback():
    """``TickEnv.session_id`` defaults to 'default'.

    ``process_tick`` previously derived the hydration seed via
    ``getattr(..., "id", "default")``; preserving that default keeps
    trajectories identical when no session is attached.
    """
    assert TickEnv(tick=1, minutes_per_tick=1.0, elapsed_minutes=1.0, race_distance_mi=1.0).session_id == "default"


def test_station_seeding_uses_marker_salt():
    """Each station gets an independent RNG sub-stream."""
    assert runner_seed("s", 1) != runner_seed("s", 2)
