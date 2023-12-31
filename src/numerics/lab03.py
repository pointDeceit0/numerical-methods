import matplotlib.pyplot as plt
import numpy as np  
import math
from lab02 import divdif
from helper import print_funcan, init_args_1, init_args_2, init_values_1, init_values_2, x0, wn, f


def interpolation_newton(init_args: list[float], init_values: list[float], interpol_args: list[float]) -> list[float]:
    '''
    Finds values of function using Newton polynomial

    Ln(x) = f(x1) + f(x1, x2)w1(x) + ... + f(x1, ..., xn)wn-1(x)
    '''
    y = []
    for x in interpol_args:
        res = init_values[0]
        for i in range(1, len(init_args)):
            res += divdif(init_args[:i + 1], init_values[:i + 1]) * wn(init_args[:i], x)
        y.append(res)
    return y


def main():
    h = 0.01
    x_1 = np.arange(min(init_args_1), max(init_args_1), h)
    x_2 = np.arange(min(init_args_2), max(init_args_2), h)

    y_1 = interpolation_newton(init_args_1, init_values_1, x_1)
    y_2 = interpolation_newton(init_args_2, init_values_2, x_2)

    err_1 = max(abs(f(x_1[i]) - y_1[i]) for i in range(len(x_1)))
    err_2 = max(abs(f(x_2[i]) - y_2[i]) for i in range(len(x_2)))
    print(err_1, err_2)
    
    print_funcan(x_1, y_1)
    print_funcan(x_2, y_2)


if __name__ == "__main__":
    main()