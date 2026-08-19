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

"""Guard tests for the LLM runner's implicit contracts.

The LLM runner path is **not exercised by the existing test suite** (no
model is called in unit tests), which makes it the highest-risk surface
during the scenario-generalization refactor: it can break at runtime
while every deterministic test stays green.

Three contracts are pinned here, each corresponding to a real failure
mode:

1. **Dynamic-instruction placeholders.** ``RUNNER_DYNAMIC_INSTRUCTION``
   interpolates session-state keys via ADK's ``inject_session_state``.
   If a refactor renames a state key (e.g. ``water`` -> ``hydration``),
   the LLM runner raises ``KeyError`` on its first tick -- at runtime,
   in production, with no test coverage. The source comment claims
   "KeyError risk is zero" *because* ``initialize_runner`` sets all
   five; this test makes that claim enforceable.

2. **Tool-schema required-ness of ``inner_thought``.** Small models
   routinely drop optional arguments. The codebase deliberately gives
   ``inner_thought`` no default so ADK marks it required. (Also covered
   in ``agents/runner/tests/test_agent.py``; re-asserted here because
   the scenario refactor touches ``process_tick``'s signature.)

3. **Static-instruction / tool-name coherence.** The prompt instructs
   the model to call ``process_tick`` by name. If the tool is renamed
   without updating the prompt, the model calls a nonexistent tool.
"""

import inspect
import re

from agents.runner.agent import (
    RUNNER_DYNAMIC_INSTRUCTION,
    RUNNER_STATIC_INSTRUCTION,
)
from agents.runner.initialization import initialize_runner
from agents.runner.running import process_tick

# Matches ADK's ``{placeholder}`` syntax. Excludes ``{{escaped}}`` braces.
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")


def test_dynamic_instruction_placeholders_are_initialized():
    """Every {var} in the dynamic instruction must exist after init.

    This is the guard against silent LLM breakage from a state-key
    rename during the scenario refactor.
    """
    placeholders = set(_PLACEHOLDER_RE.findall(RUNNER_DYNAMIC_INSTRUCTION))
    assert placeholders, "expected at least one {placeholder}; did the instruction format change?"

    state: dict = {}
    initialize_runner(state, "guard-test-session", runner_count=10)

    missing = sorted(p for p in placeholders if p not in state)
    assert not missing, (
        f"RUNNER_DYNAMIC_INSTRUCTION references state keys that "
        f"initialize_runner does not set: {missing}. The LLM runner will "
        f"raise KeyError at runtime. Either set these keys in "
        f"initialize_runner or update the instruction template."
    )


def test_dynamic_instruction_placeholder_set_is_pinned():
    """Pin the exact placeholder set so additions are a conscious act.

    Adding a placeholder is fine -- but it must be paired with a state
    key, and this test makes that pairing explicit at review time.
    """
    expected = {
        "distance",
        "water",
        "velocity",
        "runner_status",
        "target_finish_minutes",
    }
    actual = set(_PLACEHOLDER_RE.findall(RUNNER_DYNAMIC_INSTRUCTION))
    assert actual == expected, (
        f"dynamic-instruction placeholders changed: added={sorted(actual - expected)}, "
        f"removed={sorted(expected - actual)}. Update this test *and* verify "
        f"initialize_runner sets every key."
    )


def test_inner_thought_has_no_default():
    """``inner_thought`` must stay schema-required for small models."""
    sig = inspect.signature(process_tick)
    param = sig.parameters["inner_thought"]
    assert param.default is inspect.Parameter.empty, (
        "inner_thought must have no default value; ADK marks defaulted "
        "params optional and small models (gemma4:e2b) then drop them."
    )


def test_static_instruction_names_the_tick_tool():
    """The prompt must reference the actual tool function name."""
    assert process_tick.__name__ in RUNNER_STATIC_INSTRUCTION, (
        f"RUNNER_STATIC_INSTRUCTION does not mention '{process_tick.__name__}'. "
        f"If the tool was renamed, the prompt must be updated or the model "
        f"will call a nonexistent tool."
    )


def test_static_instruction_documents_required_tick_args():
    """Args the prompt promises to supply must exist in the signature."""
    sig = inspect.signature(process_tick)
    for arg in ("tick", "minutes_per_tick", "elapsed_minutes", "race_distance_mi", "collector_buffer_key"):
        assert arg in sig.parameters, f"process_tick lost parameter '{arg}' that the prompt still requires"
        assert f"`{arg}`" in RUNNER_STATIC_INSTRUCTION, (
            f"process_tick takes '{arg}' but the static instruction no longer documents it"
        )
