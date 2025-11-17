"""General utility functions for the project.

This module currently exposes a simple `add` function that returns the sum of
two inputs using Python's `+` operator.
"""

from typing import TypeVar

T = TypeVar("T")


def add(a: T, b: T) -> T:
    """Return the result of ``a + b``.

    Works for numbers and any types that implement the ``+`` operator
    (e.g., ``int``, ``float``, ``Decimal``, ``str``, lists, etc.).

    Examples:
        >>> add(2, 3)
        5
        >>> add(1.5, 2.5)
        4.0
        >>> add("foo", "bar")
        'foobar'
    """

    return a + b

