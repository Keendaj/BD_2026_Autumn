import json
import urllib.parse
import urllib.request

import numpy as np
import matplotlib.pyplot as plt

def prony_method(x, m, dt):
    size = len(x)
    X = np.zeros((size - m, m), dtype=complex)
    b = np.zeros(size - m, dtype=complex)
    for i in range(m, size):
        for k in range(1, m + 1):
            X[i - m, k - 1] = x[i - k]
        b[i - m] = -x[i]

    a, _, _, _ = np.linalg.lstsq(X, b, rcond=None)

    poly_coeffs = np.ones(m + 1, dtype=complex)
    poly_coeffs[1:] = a

    roots = np.roots(poly_coeffs)
    lambdas = np.log(np.abs(roots)) / dt
    omegas = np.arctan2(roots.imag, roots.real) / (2 * np.pi * dt)

    Z = np.zeros((size, m), dtype=complex)
    for i in range(size):
        Z[i, :] = roots ** i

    h, _, _, _ = np.linalg.lstsq(Z, x, rcond=None)
    amplitudes = np.abs(h)
    phases = np.arctan2(h.imag, h.real)

    return lambdas, omegas, amplitudes, phases

def prony_restore(lambdas, omegas, amplitudes, phases, dt, size):
    t = np.arange(size) * dt
    return sum(
        A * np.exp(l * t + 1j * (2 * np.pi * w * t + p))
        for l, w, A, p in zip(lambdas, omegas, amplitudes, phases)
    )

def print_task_number(number: int) -> None:
    print("="*30, end="")
    print(f"Task{number}", end="")
    print("="*30)

def task_1():
    print_task_number(1)
    x = np.random.normal(0, 1, size = 195) 
    x = np.hstack((x, np.array([5, -4, 3.3, 2.99, -3])))
    x = np.sort(x, axis=0)
    print("First 10 elements: ", x[:10])
    for i in [0, 1, 2, -1, -2, -3]:
        print(f"Element {x[i]} №{i}, three sigma rule result {np.abs(x[i] - np.mean(x)) > 3 * np.var(x)}" )

    plt.boxplot(x)
    plt.legend("First task Boxplot")
    plt.ylabel("Value")

    plt.show()

def generate_task_2_model(depth, h):
    i = np.arange(1, 201)
    return sum(
        k*np.exp(-h*i/k)*np.cos(4*np.pi*k*h*i + np.pi/k) for k in range(1, depth + 1)
    )

def load_weather(city, start_date, end_date):
    query = urllib.parse.urlencode({"name": city, "count": 1, "language": "ru"})
    with urllib.request.urlopen(f"https://geocoding-api.open-meteo.com/v1/search?{query}") as response:
        place = json.load(response)["results"][0]

    query = urllib.parse.urlencode({
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": place["timezone"],
    })
    with urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{query}") as response:
        daily = json.load(response)["daily"]

    return np.array(daily["time"], dtype="datetime64[D]"), np.array(daily["temperature_2m_mean"], dtype=float)

def rescaled_range(x):
    n = np.arange(3, len(x) + 1)
    rs = np.zeros(len(n))
    for j, size in enumerate(n):
        E = np.mean(x[:size])
        s = np.std(x[:size])
        X = np.cumsum(x[:size] - E)
        R = np.max(X) - np.min(X)
        rs[j] = R / s
    return n, rs

def hurst_exponent(x):
    n, rs = rescaled_range(x)
    H, c = np.polyfit(np.log(n), np.log(rs), 1)
    return H, c, n, rs

def trend_window_mean(data, width):
    m = width // 2
    return np.array([np.mean(data[max(i-m, 0):i+m+1]) for i in range(len(data))])

def trend_window_med(data, width):
    m = width // 2
    return np.array([np.median(data[max(i-m, 0):i+m+1]) for i in range(len(data))])

def trend_exp(data, alpha):
    trend = [data[0]]
    for value in data[1:]:
        trend.append(alpha * value + (1 - alpha) * trend[-1])
    return np.array(trend)

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

def task_2():
    print_task_number(2)
    h = 0.02

    for depth in (1, 2, 3):
        x = generate_task_2_model(depth, h)
        t = h * np.arange(1, len(x) + 1)

        plt.figure(figsize=(14, 7))
        plt.suptitle(f"Depth {depth}")
        for col, m in enumerate((3, 5, 7), start=1):
            lambdas, omegas, amplitudes, phases = prony_method(x, m, h)
            restored = prony_restore(lambdas, omegas, amplitudes, phases, h, len(x))
            error = np.max(np.abs(x - restored))

            print(f"Depth {depth}, m = {m}, max |x - restored| = {error:.2e}")
            for j in np.argsort(omegas):
                print(f"  lambda = {lambdas[j]:8.4f}, omega = {omegas[j]:8.4f}, A = {amplitudes[j]:.4f}, phi = {phases[j]:8.4f}")

            plt.subplot(2, 3, col)
            plt.title(f"m = {m}: model and Prony restore, error {error:.1e}")
            plt.plot(t, x, ".", label="model")
            plt.plot(t, restored.real, label="Prony")
            plt.xlabel("t")
            plt.legend()

            plt.subplot(2, 3, col + 3)
            plt.title(f"m = {m}: Prony components")
            plt.stem(omegas, amplitudes)
            plt.xlabel("omega")
            plt.ylabel("A")

        plt.tight_layout()

    plt.show()

    
def temperature_task():
    dates, temp = load_weather("Находка", "2024-10-03", "2026-10-03")

    H, c, n, rs = hurst_exponent(temp)
    print(f"Hurst exponent: H = {H:.4f}, c = {c:.4f}")

    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Hurst exponent H = {H:.3f}")
    plt.subplot(1, 2, 1)
    plt.title("R/s(n)")
    plt.plot(n, rs, ".", label="R/s")
    plt.plot(n, np.exp(c) * n ** H, label=f"{np.exp(c):.2f} n^{H:.3f}")
    plt.xlabel("n")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("y = Hz + c")
    plt.plot(np.log(n), np.log(rs), ".", label="y = ln(R/s)")
    plt.plot(np.log(n), H * np.log(n) + c, label=f"y = {H:.3f}z + {c:.3f}")
    plt.xlabel("z = ln(n)")
    plt.legend()
    plt.tight_layout()

    amplitude = np.abs(np.fft.rfft(temp)) * 2 / len(temp)
    freq = np.fft.rfftfreq(len(temp), 1)
    main = np.argmax(amplitude[1:]) + 1
    print(f"Main period: {1 / freq[main]:.1f} days, amplitude {amplitude[main]:.2f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Amplitude spectrum")
    plt.plot(freq[1:], amplitude[1:])
    plt.xlabel("frequency, 1/day")

    plt.subplot(1, 2, 2)
    plt.title(f"Low frequencies, main period {1 / freq[main]:.1f} days")
    low = freq <= 0.05
    plt.stem(freq[low][1:], amplitude[low][1:])
    plt.xlabel("frequency, 1/day")
    plt.tight_layout()

    smoothings = (
        ("Window mean", trend_window_mean, (7, 31, 91), "width"),
        ("Window median", trend_window_med, (7, 31, 91), "width"),
        ("Exponential", trend_exp, (0.3, 0.1, 0.03), "alpha"),
    )

    plt.figure(figsize=(14, 10))
    for i, (name, trend_func, params, param_name) in enumerate(smoothings):
        trend_ax = plt.subplot(3, 2, 2 * i + 1, title=f"{name}: trend")
        trend_ax.plot(dates, temp, ".", markersize=3, label="temperature")
        residuals_ax = plt.subplot(3, 2, 2 * i + 2, title=f"{name}: residuals")

        for param in params:
            trend = trend_func(temp, param)
            residuals = temp - trend
            residuals_H, _, _, _ = hurst_exponent(residuals)
            print(f"{name}, {param_name} {param}: residuals mean {np.mean(residuals):.3f}, std {np.std(residuals):.3f}, H {residuals_H:.3f}")
            check_random(residuals)

            trend_ax.plot(dates, trend, label=f"{param_name} {param}")
            residuals_ax.plot(dates, residuals, linewidth=0.8, label=f"{param_name} {param}")

        trend_ax.legend()
        residuals_ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(4)
    #task_1()
    #task_2()
    temperature_task()
    