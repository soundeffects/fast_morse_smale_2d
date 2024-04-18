from numpy import exp, pi, sin, cos, sqrt

# Definitions for functions given (x, y) coordinates, used for testing in place of images.

def monkey_saddle(x, y):
    return (x * -1.5) ** 3 - 3.0 * (x * -1.5) * (y * -1.5) ** 2

def four_peaks(x, y):
    x = x * 0.5 + 0.5
    y = y * 0.5 + 0.5
    return 0.5 * (exp(-(x - 0.25) ** 2 / 0.3 ** 2) + exp(-(y - 0.25) ** 2 / 0.3 ** 2) + exp(-(x - 0.75) ** 2 / 0.1 ** 2) + exp(-(y - 0.75) ** 2 / 0.1 ** 2))

def saddle(x, y):
    return (x * 1.5) ** 4 - 2.0 * (x * 1.5) ** 2 + (y * 1.5) ** 2

def sine(x, y):
    return sin(x * pi) + cos(y * pi)

def ripple(x, y):
    v = sqrt((x * pi ** 3) ** 2 + (y * pi ** 3) ** 2)
    epsilon = 1.0e-6
    return sin(v) / (epsilon + v)

def weak_global_minimum(x, y):
    return (x + y) ** 2

function_map = {
    'four_peaks': four_peaks,
    'monkey_saddle': monkey_saddle,
    'ripple': ripple,
    'saddle': saddle,
    'sine': sine,
    'weak_global_minimum': weak_global_minimum
}
