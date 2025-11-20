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


# Example usage
if __name__ == "__main__":
    # Test the function
    result = add(3, 5)
    print(f"3 + 5 = {result}")

    result = add(10, 20)
    print(f"10 + 20 = {result}")

    result = add(-5, 15)
    print(f"-5 + 15 = {result}")

    result = add(3.5, 2.7)
    print(f"3.5 + 2.7 = {result}")
