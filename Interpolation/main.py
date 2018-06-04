import math
import numpy as np
from matplotlib import pyplot as plt


from interpolation import *


def runge(x: float) -> float:
    return 1 / (1 + 25 * x * x)


def sincos0(x: float) -> float:
    if x < 0:
        return math.sin(math.pi * x)
    elif 0 <= x < 1 / 2:
        return math.cos(math.pi * x)
    else:
        return 0

if __name__ == '__main__':
    func = sincos0
    methods = [
        (newton_interpolation,
            np.linspace(-1, 1, 20), 'Newton'),
        (lagrange_interpolation,
            [math.cos((2 * i + 1) * math.pi / 42) for i in range(21)],  'Lagrange'),
        (piecewise_linear_interpolation,
            np.linspace(-1, 1, 20), 'Linear'),
        (natural_cubic_spline_interpolation,
            np.linspace(-1, 1, 20), 'Cubic Spline')
    ]

    xs = list(np.linspace(-1, 1, 1000))
    ys = list(map(func, xs))

    plt.plot(xs, ys, label='Original')

    for method in methods:
        sample_xs = list(method[1])
        sample_ys = list(map(func, sample_xs))
        ys = method[0](sample_xs, sample_ys, xs)
        plt.plot(xs, ys, label=method[2], linewidth=1)

    plt.legend()
    plt.show()
