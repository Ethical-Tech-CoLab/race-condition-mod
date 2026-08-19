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

import json
import logging

from google.adk.tools.tool_context import ToolContext

from agents.utils.redis_pool import get_shared_redis_client

from agents.runner.kernel import TickEnv, step

logger = logging.getLogger(__name__)


async def accelerate(intensity: float, tool_context: ToolContext) -> dict:
    """Increase the runner's speed based on the specified intensity.

    Speed is reduced when dehydrated: at 0% water, max speed is halved.

    Args:
        intensity: Acceleration intensity (0.0 to 1.0).
        tool_context: ADK tool context for state management.
    """
    velocity = tool_context.state.get("velocity", 0.0)
    water = tool_context.state.get("water", 100.0)
    crowd_responsiveness = tool_context.state.get("crowd_responsiveness", 0.5)
    boost_multiplier = 0.1
    # Dehydration penalty: linearly scale from 1.0 (full water) to 0.5 (no water)
    hydration_factor = 0.5 + 0.5 * (water / 100.0)
    new_velocity = velocity + (intensity * boost_multiplier * hydration_factor * crowd_responsiveness)
    tool_context.state["velocity"] = new_velocity
    logger.info(f"Runner accelerating: velocity={new_velocity:.2f} (hydration: {water:.0f}%)")
    return {
        "status": "success",
        "message": f"Accelerated to velocity={new_velocity:.2f} (hydration: {water:.0f}%).",
        "velocity": new_velocity,
    }


async def brake(intensity: float, tool_context: ToolContext) -> dict:
    """Decrease the runner's speed.

    Args:
        intensity: Braking intensity (0.0 to 1.0).
        tool_context: ADK tool context for state management.
    """
    velocity = tool_context.state.get("velocity", 0.0)
    new_velocity = max(0.0, velocity - (intensity * 1.5))
    tool_context.state["velocity"] = new_velocity
    logger.info(f"Runner braking: velocity={new_velocity:.2f}")
    return {
        "status": "success",
        "message": f"Braked to velocity={new_velocity:.2f}.",
        "velocity": new_velocity,
    }


async def get_vitals(tool_context: ToolContext) -> dict:
    """Get current runner vitals including speed, distance, hydration, and status.

    Args:
        tool_context: ADK tool context for state management.
    """
    vitals = {
        "status": "success",
        "message": "Vitals retrieved successfully.",
        "velocity": tool_context.state.get("velocity", 0.0),
        "distance": tool_context.state.get("distance", 0.0),
        "water": tool_context.state.get("water", 100.0),
        "exhausted": tool_context.state.get("exhausted", False),
        "collapsed": tool_context.state.get("collapsed", False),
    }
    logger.info(f"Runner vitals: {vitals}")
    return vitals


async def process_tick(
    tool_context: ToolContext,
    inner_thought: str,
    minutes_per_tick: float = -1.0,
    elapsed_minutes: float = -1.0,
    race_distance_mi: float = -1.0,
    tick: int = -1,
    collector_buffer_key: str = "",
) -> dict:
    """Advance the simulation by one tick and record your inner thought.

    Call this once per tick after setting your speed with accelerate or brake.
    Timing parameters (tick number, elapsed time, race distance) are provided
    automatically by the simulation -- you only need to supply inner_thought.

    ``inner_thought`` has no default value: ADK's auto-generated tool schema
    marks it as required, which materially improves small-model (e.g.
    gemma4:e2b) tool-call reliability. Optional fields are routinely
    dropped by 2-3B models.

    Args:
        tool_context: ADK tool context for state management.
        inner_thought: Short internal monologue (5 words max) about what
            the runner is thinking RIGHT NOW. Required (no default) so
            ADK marks it required in the tool schema -- small models
            reliably honor required fields and routinely drop optional
            ones. The autopilot path passes an empty string.
        minutes_per_tick: (Auto-provided) Simulated minutes per tick.
        elapsed_minutes: (Auto-provided) Total elapsed simulated time.
        race_distance_mi: (Auto-provided) Race distance in miles.
        tick: (Auto-provided) Current tick number.
        collector_buffer_key: (Auto-provided) Redis key for telemetry.

    Returns:
        Dict with runner vitals: distance, velocity, water, status.
    """
    state = tool_context.state

    # Read tick params from state if the caller didn't provide them.
    # Sentinel detection: ``-1`` / ``-1.0`` means "not provided." The
    # sentinel-default approach (vs. ``T | None = None``) is required to
    # work around an ADK schema-builder bug where union-with-None
    # parameter types cause ``required`` to be cleared on the entire tool
    # schema -- which lets small models (gemma4:e2b) drop the
    # ``inner_thought`` arg too. See Task I in
    # docs/plans/2026-04-19-llm-runner-cap-task-i-required-inner-thought.md.
    tick_params = state.get("_tick_params", {})
    if tick < 0:
        tick = tick_params.get("tick")
    if minutes_per_tick < 0:
        minutes_per_tick = tick_params.get("minutes_per_tick")
    if elapsed_minutes < 0:
        elapsed_minutes = tick_params.get("elapsed_minutes")
    if race_distance_mi < 0:
        race_distance_mi = tick_params.get("race_distance_mi")
    if not collector_buffer_key:
        collector_buffer_key = tick_params.get("collector_buffer_key", "")

    # Validate we have all required params from either source.
    # Return an error dict (not raise) so the LLM can recover if it calls
    # process_tick before any tick event has been received (e.g. during spawn).
    if tick is None or minutes_per_tick is None or elapsed_minutes is None or race_distance_mi is None:
        logger.warning("process_tick called without tick params -- no tick event received yet")
        return {
            "status": "error",
            "message": (
                "No tick event received yet. Wait for the simulation to send a tick event before calling process_tick."
            ),
        }

    # --- Physics: delegate to the pure kernel ---
    # The kernel owns all dynamics (velocity degradation, distance,
    # hydration, exhaustion/collapse, finish detection) and mutates
    # ``state`` in place. This function keeps only the ADK/Redis
    # plumbing. See agents/runner/kernel.py.
    env = TickEnv(
        tick=tick,
        minutes_per_tick=minutes_per_tick,
        elapsed_minutes=elapsed_minutes,
        race_distance_mi=race_distance_mi,
        session_id=getattr(getattr(tool_context, "session", None), "id", "default"),
    )
    result = step(state, env, inner_thought)

    # --- Direct-write to collector buffer (bypass PubSub bottleneck) ---
    if collector_buffer_key:
        try:
            r = get_shared_redis_client()
            if r is not None:
                session_id = getattr(getattr(tool_context, "session", None), "id", "")
                direct_msg = json.dumps(
                    {
                        "session_id": session_id,
                        "payload": {
                            "tool_name": "process_tick",
                            "result": result,
                        },
                    },
                    default=str,
                )
                await r.rpush(collector_buffer_key, direct_msg)  # type: ignore[misc]
                await r.expire(collector_buffer_key, 7200)
        except Exception:
            logger.warning(
                "process_tick: direct-write RPUSH failed for %s",
                collector_buffer_key,
                exc_info=True,
            )

    return result
