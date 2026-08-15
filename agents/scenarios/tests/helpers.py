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

"""Shared helpers for scenario tests.

``select_goal`` deliberately returns ``Destination | None`` -- "nowhere
left to go" is a meaningful outcome in an evacuation, not an error. That
makes every call site an ``Optional``, so tests that expect a
destination must narrow it.

:func:`require` does that narrowing *and* asserts the expectation, so a
regression surfaces as a clear message rather than an
``AttributeError`` on ``None``.
"""

from typing import TypeVar

T = TypeVar("T")


def require(value: T | None, what: str = "a value") -> T:
    """Assert ``value`` is not None and narrow its type.

    Args:
        value: The optional to unwrap.
        what: Description used in the failure message.

    Returns:
        ``value``, typed as non-optional.
    """
    assert value is not None, f"expected {what}, got None"
    return value
