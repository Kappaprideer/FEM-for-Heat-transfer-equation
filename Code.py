import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def B_u_v(u_der, v_der, u, v, start, end):
    return quad(lambda x: u_der(x) * v_der(x), start, end)[0] - (u(0) * v(0))


def L_v(v, start, end):
    return quad(lambda x: ((100 * x) / (x + 1)) * v(x), start, 1)[0] + quad(lambda x: 50 * v(x), 1, end)[0] - (20 * v(0))


def e(i, n):
    length = 2.0 / n
    x1 = length * (i - 1)
    x2 = length * i
    x3 = length * (i + 1)
    return lambda x: (x - x1) / (x2 - x1) if x2 >= x > x1 \
        else (x3 - x) / (x3 - x2) if x3 >= x > x2 \
        else 0.0


def D_e(i, n):
    length = 2.0 / n
    x1 = length * (i - 1)
    x2 = length * i
    x3 = length * (i + 1)
    return lambda x: 1 / (x2 - x1) if x2 >= x >= x1 \
        else -1 / (x3 - x2) if x3 >= x > x2 \
        else 0.0


if __name__ == "__main__":
    print("Number of finite elements: ", end='')
    n = int(input())

    left = np.zeros((n + 1, n + 1))
    for i in range(0, n):
        for j in range(0, n + 1):
            if abs(i - j) > 1:
                continue
            if abs(i - j) == 1:
                start = 2.0 * max(0.0, min(i, j) / n)
                end = 2.0 * min(1.0, max(i, j) / n)
            else:
                start = 2.0 * max(0.0, (i - 1) / n)
                end = 2.0 * max(1.0, (j + 1) / n)
            left[i][j] = B_u_v(D_e(j, n), D_e(i, n), e(j, n), e(i, n), start, end)
    left[n][n] = 1

    right = np.zeros(n + 1)
    for i in range(0, n):
        right[i] = L_v(e(i, n), max(0.0, (i - 1) * (2.0 / n)), min(2.0, (i + 1) * (2.0 / n)))

    u = np.linalg.solve(left, right)
    uX = np.arange(0.0, 2.0001, 2.0 / n)
    uY = np.zeros(n+1)

    for i in range(0, n + 1):
        for j in range(0, n):
            uY[i] += u[j] * e(j, n)(uX[i])

    plt.plot(uX, uY)
    plt.show()
