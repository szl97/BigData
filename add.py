def add(a, b):
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


if __name__ == "__main__":
    # Example usage
    result = add(3, 5)
    print(f"3 + 5 = {result}")

    result = add(10, 20)
    print(f"10 + 20 = {result}")

    result = add(-5, 8)
    print(f"-5 + 8 = {result}")
