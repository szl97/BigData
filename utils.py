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
    print(f"add(3, 5) = {add(3, 5)}")
    print(f"add(-1, 10) = {add(-1, 10)}")
    print(f"add(3.5, 2.5) = {add(3.5, 2.5)}")
