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

"""Tests for persona-pooled narrative.

Covers determinism (the property that makes pooling a valid substitute
for per-agent inference) and the review gate that keeps the evacuation
corpus from rendering before sign-off.
"""

import random

from agents.scenarios.narrative import (
    ALL_POOLS,
    DUMBO_POOL,
    MARATHON_POOL,
    MARIUPOL_POOL,
    NarrativePhase,
    NarrativePool,
    ResourceBucket,
    derive_bucket,
    derive_phase,
    pool_for,
    thought_for,
    thought_for_state,
)

# --- Phase and bucket derivation --------------------------------------


def test_phase_tracks_progress():
    assert derive_phase(0.0, 10.0) is NarrativePhase.START
    assert derive_phase(1.0, 10.0) is NarrativePhase.EARLY
    assert derive_phase(5.0, 10.0) is NarrativePhase.MIDDLE
    assert derive_phase(9.0, 10.0) is NarrativePhase.LATE


def test_arrival_overrides_progress():
    assert derive_phase(0.0, 10.0, arrived=True) is NarrativePhase.ARRIVED


def test_zero_length_course_is_start():
    assert derive_phase(0.0, 0.0) is NarrativePhase.START


def test_resource_buckets():
    assert derive_bucket(100.0) is ResourceBucket.FRESH
    assert derive_bucket(60.0) is ResourceBucket.OK
    assert derive_bucket(30.0) is ResourceBucket.LOW
    assert derive_bucket(5.0) is ResourceBucket.CRITICAL


# --- Fallback chain ---------------------------------------------------


def test_specific_persona_phase_wins():
    pool = NarrativePool(
        scenario_id="t",
        reviewed=True,
        entries={
            ("a", NarrativePhase.LATE): ("specific",),
            ("a", None): ("persona",),
            (None, NarrativePhase.LATE): ("phase",),
            (None, None): ("generic",),
        },
    )
    assert pool.candidates("a", NarrativePhase.LATE) == ("specific",)


def test_falls_back_through_the_chain():
    pool = NarrativePool(
        scenario_id="t",
        reviewed=True,
        entries={
            ("a", None): ("persona",),
            (None, NarrativePhase.LATE): ("phase",),
            (None, None): ("generic",),
        },
    )
    assert pool.candidates("a", NarrativePhase.LATE) == ("persona",)
    assert pool.candidates("b", NarrativePhase.LATE) == ("phase",)
    assert pool.candidates("b", NarrativePhase.EARLY) == ("generic",)


def test_missing_everything_yields_no_candidates():
    assert NarrativePool(scenario_id="t", reviewed=True).candidates("a", NarrativePhase.EARLY) == ()


# --- Determinism ------------------------------------------------------


def test_same_inputs_give_the_same_thought():
    a = thought_for(MARATHON_POOL, "run-1", tick=4, phase=NarrativePhase.MIDDLE)
    b = thought_for(MARATHON_POOL, "run-1", tick=4, phase=NarrativePhase.MIDDLE)
    assert a == b and a


def test_different_agents_get_different_thoughts():
    """Otherwise every agent would say the same thing in lockstep."""
    thoughts = {
        thought_for(MARATHON_POOL, f"run-{i}", tick=3, phase=NarrativePhase.MIDDLE) for i in range(30)
    }
    assert len(thoughts) > 1


def test_thoughts_change_over_ticks():
    thoughts = {
        thought_for(MARATHON_POOL, "run-1", tick=t, phase=NarrativePhase.MIDDLE) for t in range(20)
    }
    assert len(thoughts) > 1


def test_resource_bucket_changes_the_thought_stream():
    """Condition affects narrative without needing a separate corpus."""
    fresh = [
        thought_for(MARATHON_POOL, "r", tick=t, phase=NarrativePhase.LATE, bucket=ResourceBucket.FRESH)
        for t in range(15)
    ]
    critical = [
        thought_for(MARATHON_POOL, "r", tick=t, phase=NarrativePhase.LATE, bucket=ResourceBucket.CRITICAL)
        for t in range(15)
    ]
    assert fresh != critical


def test_selection_ignores_the_global_rng():
    """The property the golden fixture depends on."""
    random.seed(1)
    first = [thought_for(MARATHON_POOL, f"a-{i}", tick=2) for i in range(20)]
    random.seed(999)
    second = [thought_for(MARATHON_POOL, f"a-{i}", tick=2) for i in range(20)]
    assert first == second


def test_every_returned_thought_comes_from_the_corpus():
    """No generation: output is always a reviewed line."""
    corpus = set(MARATHON_POOL.all_lines())
    for i in range(50):
        for phase in NarrativePhase:
            thought = thought_for(MARATHON_POOL, f"a-{i}", tick=i, phase=phase)
            assert thought in corpus


# --- Review gate ------------------------------------------------------


def test_unreviewed_pool_renders_nothing():
    """Scenarios depicting real events stay silent until signed off."""
    assert MARIUPOL_POOL.reviewed is False
    for tick in range(10):
        for phase in NarrativePhase:
            assert thought_for(MARIUPOL_POOL, "evac-1", tick=tick, phase=phase) == ""


def test_evacuation_corpus_exists_but_is_gated():
    """The lines are authored and reviewable -- just not yet live."""
    assert MARIUPOL_POOL.all_lines()


def test_reviewed_pools_render():
    assert thought_for(MARATHON_POOL, "r", tick=1)
    assert thought_for(DUMBO_POOL, "r", tick=1, persona="commuter")


# --- Corpus quality ---------------------------------------------------


def test_thoughts_are_short_enough_for_the_hud():
    """The UI shows a single short line; long strings would clip."""
    for pool in ALL_POOLS.values():
        for line in pool.all_lines():
            assert len(line) <= 40, f"{pool.scenario_id}: '{line}' is too long for the label"
            assert len(line.split()) <= 6, f"{pool.scenario_id}: '{line}' has too many words"


def test_no_duplicate_lines_within_a_pool_entry():
    for pool in ALL_POOLS.values():
        for key, lines in pool.entries.items():
            assert len(lines) == len(set(lines)), f"{pool.scenario_id} {key} has duplicates"


def test_every_pool_covers_every_phase():
    """No situation may fall through to silence in a reviewed pool."""
    for pool in ALL_POOLS.values():
        for phase in NarrativePhase:
            assert pool.candidates(None, phase), f"{pool.scenario_id} has no lines for {phase}"


def test_dumbo_personas_have_distinct_voices():
    commuter = set(DUMBO_POOL.candidates("commuter", NarrativePhase.EARLY))
    tourist = set(DUMBO_POOL.candidates("tourist", NarrativePhase.EARLY))
    assert commuter and tourist
    assert not (commuter & tourist)


def test_evacuation_corpus_avoids_graphic_language():
    """Tone guard: practical and restrained, never sensational."""
    banned = {
        "blood", "dead", "death", "dying", "killed", "corpse", "wound",
        "scream", "bomb", "shell", "explosion", "burning", "shot",
    }
    for line in MARIUPOL_POOL.all_lines():
        words = {w.strip(".,!?'").lower() for w in line.split()}
        assert not (words & banned), f"evacuation corpus should stay restrained: '{line}'"


# --- State integration ------------------------------------------------


def test_thought_from_state_uses_progress_and_resource():
    state = {"distance": 9.5, "water": 15.0, "persona": None, "finished": False}
    assert thought_for_state(MARATHON_POOL, state, "r", tick=3, goal_distance_mi=10.0) in set(
        MARATHON_POOL.candidates(None, NarrativePhase.LATE)
    )


def test_finished_state_yields_an_arrival_thought():
    state = {"distance": 10.0, "water": 50.0, "finished": True}
    assert thought_for_state(MARATHON_POOL, state, "r", tick=9, goal_distance_mi=10.0) in set(
        MARATHON_POOL.candidates(None, NarrativePhase.ARRIVED)
    )


def test_state_integration_is_deterministic():
    state = {"distance": 3.0, "water": 80.0, "finished": False}
    a = thought_for_state(DUMBO_POOL, state, "ped-1", tick=2, goal_distance_mi=10.0)
    b = thought_for_state(DUMBO_POOL, state, "ped-1", tick=2, goal_distance_mi=10.0)
    assert a == b


# --- Lookup -----------------------------------------------------------


def test_pool_lookup_by_scenario_id():
    assert pool_for("marathon") is MARATHON_POOL
    assert pool_for("dumbo") is DUMBO_POOL


def test_unknown_scenario_gets_a_safe_empty_pool():
    unknown = pool_for("does-not-exist")
    assert unknown.reviewed is False
    assert thought_for(unknown, "r", tick=1) == ""


def test_pool_ids_match_their_keys():
    for scenario_id, pool in ALL_POOLS.items():
        assert pool.scenario_id == scenario_id
