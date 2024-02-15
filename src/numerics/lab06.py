"""
Methods of solving the Cauchy problem

Runge-Kutta method (3rd and 4th order), Euler's implicit and Explicit methods

"""
import matplotlib.pyplot as plt
import numpy as np
from helper import print_funcan, round_to


def f(x: float, y: float) -> float:
    return (y**2 * np.log(x) - y) / x

"""def f(x: float, y: float) -> float:
    return -y * np.tan(x) + np.sin(2 * x)"""


def sol(x: float) -> float:
    return 1 / (np.log(x) + 1 + x)

"""def sol(x: float) -> float:
    return -2 * (np.cos(x))**2 + np.cos(x)"""


def euler_explicit(domain: list[float], y0: float, h=None) -> list[float]:
    """
    Euler's explicit (geometric) method
        domain: numpy.arange[float], scope of definition

        y0:     float, inition value

        h:      float, if None it must be considered as unequal step

    """
    n = len(domain)
    y = [y0] + [0] * (n - 1)

    for i in range(1, n):
        y[i] = y[i - 1] + h * f(domain[i - 1], y[i - 1])
    
    return y


def euler_implicit(domain: list[float], y0: float, h=None, k=4) -> list[float]:
    """
    Euler's implicit (trapezoid) method
        domain: numpy.arange[float], scope of definition

        y0:     float, inition value

        h:      float, if None it must be considered as unequal step

        k:      int, order of avaraging 

    """ 
    n = len(domain)
    y = [y0] + [0] * (n - 1)

    for i in range(1, n):
        temp = f(domain[i - 1], y[i - 1])
        y[i] = y[i - 1] + h * temp
        t = y[i]

        for _ in range(1, k + 1):
            y[i] = y[i - 1] + h * (temp + f(domain[i], y[i])) / 2

    return y


def rkutta3(domain: list[float], y0: float, h=None) -> list[float]:
    """
    Runge-Kutta methods
        domain: numpy.arange[float], scope of definition

        y0:     float, inition value

        h:      float, if None it must be considered as unequal step

    """
    n = len(domain)
    y = [y0] + [0] * (n - 1)

    for i in range(1, n):
        k1 = f(domain[i - 1],             y[i - 1])
        k2 = f(domain[i - 1] + h / 3,     y[i - 1] + h / 3 * k1)
        k3 = f(domain[i - 1] + 2 * h / 3, y[i - 1] + 2 * h / 3 * k2)

        y[i] = y[i - 1] + h / 6 * (k1 + 4 * k2 + k3)

    return y


def rkutta4(domain: list[float], y0: float, h=None) -> list[float]:
    """
    Runge-Kutta methods
        domain: numpy.arange[float], scope of definition

        y0:     float, inition value

        h:      float, if None it must be considered as unequal step

    """
    n = len(domain)
    y = [y0] + [0] * (n - 1)

    for i in range(1, n):
        k1 = f(domain[i - 1],         y[i - 1])
        k2 = f(domain[i - 1] + h / 2, y[i - 1] + h / 2 * k1)
        k3 = f(domain[i - 1] + h / 2, y[i - 1] + h / 2 * k2)
        k4 = f(domain[i - 1] + h,     y[i - 1] + h * k3)

        y[i] = y[i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y


def errors(y: list[float], domain: list[float], name=None, func=sol, prnt=True) -> tuple[float, float, float]:
    """
    Finds absolute error (C space), relative error and norm in l2

    """
    abs_error = 0
    rel_error = 0
    for i, x in enumerate(domain):
        abs_error = max(abs_error, abs(func(x) - y[i]))
        rel_error = max(rel_error, abs(func(x) - y[i]) / abs(func(x)))
    l2_norm = sum(x**2 for x in y) ** 0.5
    if prnt:
        print(f'Absolute error of {name}:\t{round_to(abs_error)}')
        print(f'l2 norm of {name}:\t\t{round_to(l2_norm)}')
        print(f'Relative error of {name}:\t{round_to(rel_error * 100)} %\n')
    return abs_error, l2_norm, rel_error


def main(): 
    x0, y0 = 1, 0.5
    h = 0.1
    a, b = 1, 2

    '''x0, y0 = 0, -1
    h = 0.1
    a, b = x0, 1'''

    domain = np.arange(a, b + h, h)
    
    y_euler_expl = euler_explicit(domain, y0, h)
    y_euler_impl = euler_implicit(domain, y0, h)
    #print_funcan(domain, y_euler_expl, y_euler_impl, names=["Explicit", "Implicit"], func=sol)
    errors(y_euler_expl, domain, name="Euler\'s explicit method")
    errors(y_euler_impl, domain, name="Euler\'s implicit method")

    y_rk3 = rkutta3(domain, y0, h)
    y_rk4 = rkutta4(domain, y0, h)
    #print_funcan(domain, y_rk3, y_rk4, names=["Runge-Kutta 3rd order", "Runge-Kutta 4th order"], func=sol)
    errors(y_rk3, domain, name="Runge-kutta method (3rd order)")
    abs_err1, l21, rel_err1 = errors(y_rk4, domain, name="Runge-kutta method (4rd order)")

    # order defining
    h2 = 0.05
    domain = np.arange(a, b + h2, h2)
    y_rk3_2 = rkutta4(domain, y0, h2)
    abs_err2, l22, rel_err2 = errors(y_rk3_2, domain, prnt=False)
    print(abs_err2, l22, rel_err2)

    print(np.log2(abs_err1 / abs_err2), np.log2(rel_err1 / rel_err2))


if __name__ == "__main__":
    main()
