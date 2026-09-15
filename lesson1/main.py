def print_task_number(number: int) -> None:
    print("="*30, end="")
    print(f"Task{number}", end="")
    print("="*30)

def w_norm(x: list[int], weights: list[int] | None = None) -> int:
    if weights is None:
        weights = [1/len(x)] * len(x)

    if len(weights) != len(x):
        raise RuntimeError(f"Can't calculate norm with different dimensions {len(x)} vs {len(weights)}")
    
    if any(i < 0 for i in weights):
        raise RuntimeError(f"Weights must be non negative: {weights}")

    if (sum(weights) == 0):
        raise RuntimeError(f"Weights sum must be greater than zero")

    if sum(weights) != 1:
        w_sum = sum(weights)
        weights = [i / w_sum for i in weights]
        print("\n[WARNING] Weights sum not equal 1, rescaling weights:", weights)
    
    result = 0

    for i, w in zip(x, weights):
        result += i * w

    return result

def norm(x: list[int], k: int):
    if k <= 0:
        raise ValueError("K must be greater than zero")

    if int(k) != k:
        raise TypeError("K must be integer")

    result = 0
    
    for i in x:
        result += i ** k

    result **= 1/k

    return result

def inf_norm(x: list[int]):
    result = max(map(abs, x))

    return result

def factorial(n):
    result = 1

    for i in range(1, n+1):
        result*=i

    return result

def task_123():
    print_task_number(1)
    x = list(range(-10,6))
    print(f"Vector X:\n({"; ".join(map(str, x))})")
    y = list(range(-5,11))
    print(f"Vector Y:\n({"; ".join(map(str, y))})")

    print_task_number(2)
    z = [x[i] if i % 2 == 1 else y[i] for i in range(len(x))]
    print(f"Vector Z:\n({"; ".join(map(str, z))})")
    z.sort()
    print(f"Vector Z sorted:\n({"; ".join(map(str, z))})")

    print_task_number(3)
    print("Manhattan norm X:", norm(x, 1))
    print("Manhattan norm Y:", norm(y, 1))
    print("Manhattan norm Z:", norm(z, 1))
    print("\n")
    print("Euclidean norm X:", norm(x, 2))
    print("Euclidean norm Y:", norm(y, 2))
    print("Euclidean norm Z:", norm(z, 2))
    print("\n")
    print("Chebyshev norm X:", inf_norm(x))
    print("Chebyshev norm Y:", inf_norm(y))
    print("Chebyshev norm Z:", inf_norm(z))
    
def task_4():
    print_task_number(4)
    for i in range(10):
        print(f"{i}! = {factorial(i)}")

def read_vector(size: int, name: str) -> list[int]:
    result = []

    for i in range(size):
        x = int(input(f"Enter {name} {i+1} element: "))
        result.append(x)

    return result

def task_5():
    print_task_number(5)
    size = int(input("Enter vector size: "))
    x = read_vector(size, "vector")
    weights = read_vector(size, "weights")

    n = w_norm(x, weights)

    abs_x = list(map(abs, x))
    min_ele = min(abs_x)
    max_ele = max(abs_x)

    sum_x = sum(x)

    print("Minimal absolute element: ", min_ele)
    print("Maximum absolute element: ", max_ele)
    print("Sum: ", sum_x)
    print("Weighted norm: ", n)

if __name__ == "__main__":
    task_123()
    task_4()
    task_5()