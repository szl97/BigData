"""
通用工具函数。

此模块提供一个简单的加法函数 `add(a, b)`，返回 `a + b`。
"""

from typing import Any


def add(a: Any, b: Any) -> Any:
    """返回 a + b。

    参数:
        a: 任意支持 `+` 运算的对象。
        b: 与 `a` 类型兼容的对象。

    返回:
        a 与 b 使用 `+` 运算的结果。
    """
    return a + b


__all__ = ["add"]

