import numpy as np
import matplotlib.pyplot as plt

k = np.arange(0, 501)
h = 0.1

def trend_window_mean(data, width):
    m = width // 2
    return np.array([np.mean(data[i-m:i+m+1]) for i in range(m, len(data) - m)])

def trend_window_med(data, width):
    m = width // 2
    return np.array([np.median(data[i-m:i+m+1]) for i in range(m, len(data) - m)])

def turning_points(data):
    count = 0
    for i in range(1, len(data) - 1):
        if data[i-1] < data[i] > data[i+1] or data[i-1] > data[i] < data[i+1]:
            count += 1
    return count

def kendall(data):
    n = len(data)
    p = 0
    for i in range(n):
        p += np.sum(data[i+1:] > data[i])
    return 4 * p / (n * (n - 1)) - 1

def check_random(data):
    n = len(data)

    p = turning_points(data)
    p_mean = 2 * (n - 2) / 3
    p_std = np.sqrt((16 * n - 29) / 90)
    print(f"  Turning points: {p} (expected {p_mean:.1f}), random: {abs(p - p_mean) < 1.96 * p_std}")

    tau = kendall(data)
    tau_std = np.sqrt(2 * (2 * n + 5) / (9 * n * (n - 1)))
    print(f"  Kendall tau: {tau:.4f}, random: {abs(tau) < 1.96 * tau_std}")

def check_mean_and_normality(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    print(f"  Mean: {mean:.4f}, unbiased: {abs(mean) < 1.96 * std / np.sqrt(n)}")

    skew = np.mean((data - mean) ** 3) / std ** 3
    kurt = np.mean((data - mean) ** 4) / std ** 4 - 3
    normal = abs(skew) < 1.96 * np.sqrt(6 / n) and abs(kurt) < 1.96 * np.sqrt(24 / n)
    print(f"  Skewness: {skew:.3f}, kurtosis: {kurt:.3f}, normal: {normal}")

def trend_exp(data, alpha):
    trend = [data[0]]
    for value in data[1:]:
        trend.append(alpha * value + (1 - alpha) * trend[-1])
    return np.array(trend)

def print_task_number(number: int) -> None:
    print("="*30, end="")
    print(f"Task{number}", end="")
    print("="*30)

def task_1():
    print_task_number(1)
    h = 0.05
    exact = np.sqrt(k*h)
    x = exact + np.random.normal(0, 1, len(k))

    for i, name, trend_func in ((1, "Mean", trend_window_mean), (2, "Median", trend_window_med)):
        plt.subplot(1, 2, i)
        plt.title(name)
        plt.plot(k, x, ".")
        plt.plot(k, exact, label="exact")

        for width in (21, 51, 111):
            m = width // 2
            trend = trend_func(x, width)
            print(f"{name}, width {width}:")
            check_random(x[m:-m] - trend)
            plt.plot(k[m:-m], trend, label=f"width {width}")

        plt.legend()

    plt.show()

def task_2():
    print_task_number(2)
    exact = 0.5*np.sin(k*h)
    x = exact + np.random.normal(0, 1, len(k))

    amplitude = np.abs(np.fft.rfft(x)) * 2 / len(x)
    freq = np.fft.rfftfreq(len(x), h)
    print(f"Main frequency: {freq[np.argmax(amplitude)]:.4f} (exact 1/(2pi) = {1 / (2*np.pi):.4f})")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Exponential moving average")
    plt.plot(k, x, ".")
    plt.plot(k, exact, label="exact")

    for alpha in (0.01, 0.05, 0.1, 0.3):
        trend = trend_exp(x, alpha)
        print(f"alpha {alpha}:")
        check_random(x - trend)
        check_mean_and_normality(x - trend)
        plt.plot(k, trend, label=f"alpha {alpha}")

    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Amplitude spectrum")
    plt.plot(freq, amplitude)

    plt.show()

if __name__ == "__main__":
    task_1()
    task_2()
