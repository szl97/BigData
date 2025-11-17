"""通用工具函数模块。

包含简单的加法函数 `add(a, b)`，返回 a 与 b 的和。
"""

from __future__ import annotations

__all__ = ["add"]


def add(a, b):
    """返回 a 与 b 的和。

    参数:
        a: 任意支持加法运算的对象
        b: 任意支持加法运算的对象

    返回:
        a + b 的结果
    """

    return a + b


if __name__ == "__main__":
    # 简单演示：仅在直接运行时执行
    import sys

    if len(sys.argv) == 3:
        x, y = sys.argv[1], sys.argv[2]
        try:
            # 尝试将输入解析为数字，否则按字符串相加
            from decimal import Decimal

            try:
                x_val = Decimal(x)
                y_val = Decimal(y)
                # 保持最简：Decimal 的和再根据是否为整数进行展示
                result = x_val + y_val
                # 去掉可能的无意义小数位，例如 8.0 -> 8
                if result == result.to_integral_value():
                    print(int(result))
                else:
                    print(result.normalize())
            except Exception:
                print(x + y)
        except Exception:
            print(x + y)
    else:
        print("用法: python utils.py <a> <b>")

