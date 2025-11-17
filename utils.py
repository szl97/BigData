"""
通用工具函数。

函数：add(a, b) -> 返回两数之和。
"""

from numbers import Number


def add(a: Number, b: Number) -> Number:
    """
    返回 a + b。

    参数:
        a: 加数（数字类型）
        b: 加数（数字类型）

    返回:
        两数之和。

    示例:
        >>> add(1, 2)
        3
        >>> add(2.5, 0.5)
        3.0
    """

    return a + b

