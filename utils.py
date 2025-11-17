"""
通用工具函数。

提供一个简单的加法函数 `add(a, b)`，返回 a + b。
"""

from typing import Any


def add(a: Any, b: Any):
    """
    返回 a 与 b 的和（a + b）。

    说明：
    - 适用于数字相加，也适用于实现了 `__add__` 的对象（如字符串拼接）。
    - 若操作数不支持相加，将由 Python 抛出相应异常（如 TypeError）。

    参数：
    - a: 任意可与 b 相加的对象
    - b: 任意可与 a 相加的对象

    返回：
    - 相加结果
    """

    return a + b

