import matplotlib.pyplot as plt
import numpy as np  
import math
from lab01 import interpolation_lagrange
from helper import init_args_1, init_args_2, init_values_1, init_values_2, x0, f, wn


def derivative(x0: float, h: float, order: int, f=f) -> float:
    """
    Finding derivative numerically by using

    derivative(f(x(n))) = f(x(n + 1)) - f(x(n)) / delta(x) 
    """
    if order == 1:
        return (f(x0 + h) - f(x0)) / h
    return (derivative(x0 + h, h, order - 1) - derivative(x0, h, order - 1)) / h


def divdif(args: list[float], values: list[float]) -> float:
    '''
    Calculates divide difference

    f[x(j), ..., x(j+k)] = {f[x(j+1), ..., x(j+k)] - f[x(j), ..., x(j+k-1)]} / {x(j+k) - x(0)}
    '''
    if len(args) == 2: # Inition condition
        return (values[1] - values[0]) / (args[1] - args[0])
    return (divdif(args[1:], values[1:]) - divdif(args[:-1], values[:-1])) / (args[-1] - args[0])


def task_1(init_args: list[float], init_values: list[float], x0: float) -> None:
    # calculating value in x0 with Lagrange polinome
    h = 0.001 # step 
    x = np.arange(min(init_args), max(init_args), h) # [a, b]
    
    y0 = interpolation_lagrange(init_args, init_values, [x0])[0] # intrepolation value of x0

    order = len(init_args)
    
    # calculating max(abs(fprime n (x))) <= M, x from [a, b]
    M = max([derivative(x0, h, order) for x0 in x]) 

    print(f'Оценка погрешности:\n\t{abs(f(x0) - y0)} <= {M * wn(init_args, x0) / math.factorial(order)}')


def task_2(init_args: list[float], init_values, x0: float) -> None:
    init_args = [x0] + init_args
    init_values = [f(x0)] + init_values

    dd = divdif(init_args, init_values)

    y0 = interpolation_lagrange(init_args[1:], init_values[1:], [x0])[0]

    print(f'Оценка погрешности с помощью разделенных разностей:\n\t{round(f(x0) - y0, 7)} = {round(dd * wn(init_args[1:], x0), 7)}')


def main():
    task_1(init_args_1, init_values_1, x0)
    task_1(init_args_2, init_values_2, x0)

    task_2(init_args_1, init_values_1, x0)
    task_2(init_args_2, init_values_2, x0)


if __name__ == "__main__":
    main()