def add(a, b):
    """
    计算两个数的和

    参数:
        a: 第一个数
        b: 第二个数

    返回:
        a和b的和
    """
    return a + b


# 测试函数
if __name__ == "__main__":
    # 测试用例
    result1 = add(3, 5)
    print(f"add(3, 5) = {result1}")

    result2 = add(-10, 20)
    print(f"add(-10, 20) = {result2}")

    result3 = add(3.14, 2.86)
    print(f"add(3.14, 2.86) = {result3}")

    result4 = add(0, 0)
    print(f"add(0, 0) = {result4}")
