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

"""Golden-fixture regression contract for the marathon scenario.

This is the safety net for the scenario-generalization work (see
``docs/plans/`` and the Seam A–H decomposition). It pins the *exact*
numeric output of the deterministic runner pipeline so that any refactor
which changes physics, profile generation, or tick semantics fails loudly.

Why this works: runner profiles derive from ``sha256(session_id)`` via
``runner_seed``, and ``process_tick`` is pure arithmetic over session
state. Given fixed session IDs and fixed tick parameters, the entire
trajectory is reproducible bit-for-bit.

Contract
--------
``test_golden_trajectory_matches_fixture`` compares a freshly computed
trajectory against ``golden_marathon.json``. If it fails, either:

1. You introduced an unintended behavior change -> fix the code; or
2. You made a deliberate, reviewed change -> re-bless the fixture:

   ``GOLDEN_BLESS=1 pytest agents/tests/test_golden_marathon.py -k bless``

   then commit the diff **with justification**.

Re-blessing runs through pytest (not a standalone script) so the fixture
is generated under exactly the same import conditions it is verified
under -- the root ``conftest.py`` mocks GCP credentials, without which
``agents.runner`` cannot even be imported.

Never re-bless casually. The fixture is the definition of "marathon
still works".
"""

import hashlib
import json
import os
import pathlib

import pytest

from agents.runner.constants import MARATHON_MI
from agents.runner.initialization import initialize_runner
from agents.runner.running import process_tick

FIXTURE_PATH = pathlib.Path(__file__).parent / "golden_marathon.json"

# --- Frozen scenario parameters (do not change without re-blessing) ---
RUNNER_COUNT = 8
MAX_TICKS = 10
TOTAL_RACE_HOURS = 6.0
MINUTES_PER_TICK = (TOTAL_RACE_HOURS * 60) / MAX_TICKS
SESSION_IDS = [f"golden-runner-{i:03d}" for i in range(RUNNER_COUNT)]

# Fields captured per tick. Deliberately excludes ``inner_thought``
# (LLM-authored, non-deterministic) so this fixture is valid for both the
# autopilot and LLM runner paths.
CAPTURED_FIELDS = (
    "tick",
    "runner_status",
    "velocity",
    "effective_velocity",
    "distance",
    "water",
    "mi_this_tick",
    "exhausted",
    "collapsed",
    "pace_min_per_mi",
    "finish_time_minutes",
    "wave_number",
)


class _FakeToolContext:
    """Minimal ToolContext stand-in: ``state`` dict + ``session.id``.

    ``process_tick`` only touches ``.state`` and ``.session.id`` (the
    latter for per-hydration-station seeded RNG), so a real ADK context
    is unnecessary and would add import/runtime cost.
    """

    def __init__(self, state: dict, session_id: str) -> None:
        self.state = state
        self.session = type("S", (), {"id": session_id})()


async def _compute_trajectory() -> dict:
    """Run the deterministic pipeline and return the full trajectory."""
    runners: list[dict] = []

    for session_id in SESSION_IDS:
        state: dict = {}
        initialize_runner(state, session_id, RUNNER_COUNT)

        # Profile is the L0 identity layer: pinned separately so a change
        # to the distributions fails with a clear, specific diff.
        profile = {
            "target_finish_minutes": state["target_finish_minutes"],
            "velocity": state["velocity"],
            "water": state["water"],
            "will_hit_wall": state["will_hit_wall"],
            "wall_mi": state["wall_mi"],
            "wall_severity": state["wall_severity"],
            "hydration_efficiency": state["hydration_efficiency"],
            "crowd_responsiveness": state["crowd_responsiveness"],
            "wave_number": state["wave_number"],
            "start_delay_minutes": state["start_delay_minutes"],
        }

        ctx = _FakeToolContext(state, session_id)
        ticks: list[dict] = []
        for tick in range(1, MAX_TICKS + 1):
            result = await process_tick(
                tool_context=ctx,
                inner_thought="",
                minutes_per_tick=MINUTES_PER_TICK,
                elapsed_minutes=tick * MINUTES_PER_TICK,
                race_distance_mi=MARATHON_MI,
                tick=tick,
                collector_buffer_key="",
            )
            ticks.append({k: result.get(k) for k in CAPTURED_FIELDS})

        runners.append({"session_id": session_id, "profile": profile, "ticks": ticks})

    return {
        "scenario": "marathon",
        "runner_count": RUNNER_COUNT,
        "max_ticks": MAX_TICKS,
        "minutes_per_tick": MINUTES_PER_TICK,
        "race_distance_mi": MARATHON_MI,
        "runners": runners,
    }


def _digest(trajectory: dict) -> str:
    """Stable SHA-256 over the canonical JSON form of the trajectory."""
    canonical = json.dumps(trajectory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.asyncio
async def test_golden_fixture_exists():
    """The fixture must be committed; without it there is no contract."""
    assert FIXTURE_PATH.exists(), (
        f"Golden fixture missing at {FIXTURE_PATH}. Generate it with:\n"
        f"  python -m agents.tests.golden_marathon"
    )


@pytest.mark.asyncio
async def test_trajectory_is_deterministic():
    """Two runs in the same process must agree.

    Guards against hidden global-RNG usage. ``process_tick``'s station
    check must use its own seeded RNG, never ``random.random()``.
    """
    first = await _compute_trajectory()
    second = await _compute_trajectory()
    assert _digest(first) == _digest(second), "trajectory is not reproducible within a single process"


@pytest.mark.asyncio
async def test_golden_trajectory_matches_fixture():
    """THE regression contract. See module docstring before re-blessing."""
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    actual = await _compute_trajectory()

    # Compare structurally first so failures point at the exact runner/tick
    # rather than dumping an opaque hash mismatch.
    assert actual["runner_count"] == expected["runner_count"]
    assert actual["max_ticks"] == expected["max_ticks"]

    for exp_runner, act_runner in zip(expected["runners"], actual["runners"], strict=True):
        sid = exp_runner["session_id"]
        assert act_runner["session_id"] == sid
        assert act_runner["profile"] == exp_runner["profile"], f"profile drift for {sid}"
        for exp_tick, act_tick in zip(exp_runner["ticks"], act_runner["ticks"], strict=True):
            assert act_tick == exp_tick, f"tick drift for {sid} at tick {exp_tick['tick']}"

    assert _digest(actual) == _digest(expected), "digest mismatch despite structural equality"


@pytest.mark.asyncio
async def test_marathon_invariants():
    """Scenario-level sanity, independent of the pinned numbers.

    These survive re-blessing and catch nonsense (negative water, runners
    exceeding the course) that a stale fixture could otherwise enshrine.
    """
    trajectory = await _compute_trajectory()
    for runner in trajectory["runners"]:
        for tick in runner["ticks"]:
            assert 0.0 <= tick["water"] <= 100.0, "water out of range"
            assert tick["distance"] >= 0.0, "negative distance"
            assert tick["distance"] <= MARATHON_MI + 1e-9, "runner exceeded course length"
            assert tick["effective_velocity"] >= 0.0, "negative velocity"
            assert tick["runner_status"] in {"running", "exhausted", "collapsed", "finished"}

        distances = [t["distance"] for t in runner["ticks"]]
        assert distances == sorted(distances), "distance must be monotonically non-decreasing"


@pytest.mark.skipif(
    os.environ.get("GOLDEN_BLESS") != "1",
    reason="fixture regeneration is opt-in; set GOLDEN_BLESS=1 to re-bless",
)
@pytest.mark.asyncio
async def test_bless_fixture():
    """Regenerate the golden fixture. Opt-in via ``GOLDEN_BLESS=1``.

    Deliberately a test rather than a script: it inherits the root
    ``conftest.py`` GCP credential mock, so generation and verification
    share identical import conditions.
    """
    trajectory = await _compute_trajectory()
    FIXTURE_PATH.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {FIXTURE_PATH}\ndigest: {_digest(trajectory)}")
