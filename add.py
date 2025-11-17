def add(a, b):
    """
    计算两个数的和

    参数:
        a: 第一个数
        b: 第二个数

    返回:
        a + b 的结果
    """
    return a + b


# 示例用法
if __name__ == "__main__":
    result = add(3, 5)
    print(f"3 + 5 = {result}")

    result = add(10.5, 20.3)
    print(f"10.5 + 20.3 = {result}")
