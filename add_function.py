def add(a, b):
    """
    返回两个数的和

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
    print(f"add(3, 5) = {result1}")

    result2 = add(10.5, 20.3)
    print(f"add(10.5, 20.3) = {result2}")

    result3 = add(-5, 8)
    print(f"add(-5, 8) = {result3}")
