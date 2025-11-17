def add(a, b):
    """
    计算两个数的和

    参数:
        a: 第一个数
        b: 第二个数

    返回:
        a + b 的和
    """
    return a + b


# 测试示例
if __name__ == "__main__":
    # 测试整数
    result1 = add(3, 5)
    print(f"add(3, 5) = {result1}")

    # 测试浮点数
    result2 = add(3.5, 2.7)
    print(f"add(3.5, 2.7) = {result2}")

    # 测试负数
    result3 = add(-10, 5)
    print(f"add(-10, 5) = {result3}")
