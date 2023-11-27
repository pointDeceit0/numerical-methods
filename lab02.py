import matplotlib.pyplot as plt
import numpy as np  
import math
from lab01 import print_func, interpolation_lagrange


def func(x: float): # initial function
    return np.arccos(x)


def derivative(x0: float, h: float, order: int, f=func) -> float:
    """
    Finding derivative numerically by using

    derivative(f(x(n))) = f(x(n + 1)) - f(x(n)) / delta(x) 
    """
    if order == 1:
        return (f(x0 + h) - f(x0)) / h
    return (derivative(x0 + h, h, order - 1) - derivative(x0, h, order - 1)) / h


def find_w(init_args: list[float], x0: float) -> float:
    '''
    Calculates value of wn

    wn(x) = mul(x0 - xj) for j = 1 to n
    '''
    wn = 1
    for x in init_args:
        wn *= abs(x0 - x)
    return wn


def divdif(args: list[float], values: list[float]) -> float:
    '''
    Calculates divide difference

    f[x(j), ..., x(j+k)] = {f[x(j+1), ..., x(j+k)] - f[x(j), ..., x(j+k-1)]} / {x(j+k) - x(0)}
    '''
    if len(args) == 2: # Inition condition
        return (values[1] - values[0]) / (args[1] - args[0])
    return (divdif(args[1:], values[1:]) - divdif(args[:-1], values[:-1])) / (args[-1] - args[0])


def task_1(init_args: list[float], x0: float) -> None:
    # calculating value in x0 with Lagrange polinome
    h = 0.001 # step 
    x = np.arange(min(init_args), max(init_args), h) # [a, b]
    
    y0 = interpolation_lagrange(init_args, [func(v) for v in init_args], [x0])[0] # intrepolation value of x0

    order = len(init_args)
    
    # calculating max(abs(fprime n (x))) <= M, x from [a, b]
    M = max([derivative(x0, h, order) for x0 in x]) 

    print(f'Оценка погрешности:\n\t{abs(func(x0) - y0)} <= {M * find_w(init_args, x0) / math.factorial(order)}')


def task_2(init_args: list[float], x0: float) -> None:
    init_args = [x0] + init_args
    init_values = [func(x) for x in init_args]

    dd = divdif(init_args, init_values)

    y0 = interpolation_lagrange(init_args[1:], init_values[1:], [x0])[0]

    print(f'Оценка погрешности с помощью разделенных разностей:\n\t{round(func(x0) - y0, 7)} = {round(dd * find_w(init_args[1:], x0), 7)}')


def main():
    x0 = 0.1
    init_args_1 = [-0.4, -0.1, 0.2, 0.5]
    init_args_2 = [-0.4, 0, 0.2, 0.5]

    task_1(init_args_1, x0)
    task_1(init_args_2, x0)

    task_2(init_args_1, x0)
    task_2(init_args_2, x0)

    #x1 = 1
    #init_args_3 = [0, np.pi / 6, 2*np.pi / 6, 3*np.pi / 6]
    #task_2(init_args_3, x1)


if __name__ == "__main__":
    main()