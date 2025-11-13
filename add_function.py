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


if __name__ == "__main__":
    # 测试示例
    result1 = add(3, 5)
    print(f"3 + 5 = {result1}")

    result2 = add(10.5, 2.3)
    print(f"10.5 + 2.3 = {result2}")

    result3 = add(-7, 15)
    print(f"-7 + 15 = {result3}")
