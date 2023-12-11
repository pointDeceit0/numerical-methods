import matplotlib.pyplot as plt
import numpy as np  
from helper import *
from lab02 import divdif
from random import randint


def tdm(x: list[float], y: list[float]) -> list[list[float]]:
    ''' 
    Tridiagonal matrix algorithm or Thomas algorithm. 
        x : standard list, interpolation nodes
        
        y : stadard list, values of interpolation nodes

    Is used to find values of matrix. Is a simplified form of Gaussian elimination
    and unlike of Gaussian method (O(n^3)) has O(n) time complexity


    Of course, there could be done a lot of optimisations:
    1. The most significant - not use python. Same algorithms could be written with e.g. c++ and it will give a huge boost
    2. Caching divided difference. In this case it isn't to crucial, but generally speaking - it does
    3. Also it would be better if possible user errors were taken into account

    (*) I'm not sure, but there's particular cases with cubic splines, where could be written particular formulas for this cases.
        It can give a boost either.
    (**) However, this programm describes the main idea of algorithm, in future it would be better use existing functions from libraries
    '''
    n = len(x)
    fs = [[0, 0] for _ in range(n - 1)] # forward sweep: [[delta_0, lambda_0], ..., [delta_{n-1}, lambda_{n-1}]]

    # initial case for delta and lambda
    fs[0][0] = -(x[2] - x[1]) / (x[2] - x[0])
    fs[0][1] = 3 * (divdif(x[1:3], y[1:3]) - divdif(x[0:2], y[0:2])) / (2*x[2] - 2*x[0])
  
    for k in range(2, n):
        d = 2*(x[k] - x[k-2]) + (x[k-1] - x[k-2]) * fs[k-2][0] # denominator is equial for delta and lambda

        fs[k-1][0] = -(x[k] - x[k-1]) / d
        fs[k-1][1] = (3*(divdif(x[k-1:k+1], y[k-1:k+1]) - divdif(x[k-2:k], y[k-2:k])) - (x[k-1] - x[k-2]) * fs[k-2][1]) / d

    bs = [0] * (n - 1) # backward substitution: [coef_1, ..., coef_{n-1}]
    bs[-1] = fs[-1][1] # initial case, when delta = 0 => x_n = lambda_{n-1}

    for k in range(n - 3, -1, -1):
       bs[k-1] = fs[k-1][0] * bs[k] + fs[k-1][1] 

    return bs
        

def cuspline(init_args: list[float], init_values: list[float], x: list[float]) -> list[float]:
    '''
    Cubic spline of defect 1
        init_args: standard list, interpolation nodes

        init_values: standard list, values of interpolation nodes

        x: np.arrange, scope of definition

    Cubic splines of defect 1 allows to find values of unknown function with 
    interpolation nodes. Builds cubic polynomials on every sector between two 
    adjacent nodes. It has O(n), it must be taken in account how many constant operations
    lies in it.

    Polinomial view: S(x) = a + b(x - x0) + c(x - x0)^2 + d(x - x0)^3

    Also could be done a lot of optimisations, which almost the same as for tdm above.
    '''
    n = len(init_args)
    # coefficients
    c = tdm(init_args, init_values)
    d = [0] * (n - 1)
    b = [0] * (n - 1)
    
    # initial cases, where imagine c[0 - 1] = 0
    d[0] = c[0] / (3 * (init_args[1] - init_args[0])) 
    b[0] = divdif(init_args[0:2], init_values[0:2]) + 2 / 3 * c[0] * (init_args[1] - init_args[0])

    for i in range(1, n - 1):
        d[i] = (c[i] - c[i-1]) / (3 * (init_args[i+1] - init_args[i])) 
        b[i] = divdif(init_args[i:i+2], init_values[i:i+2]) + 2 / 3 * c[i] * (init_args[i+1] - init_args[i]) + 1 / 3 * c[i-1] * (init_args[i+1] - init_args[i])

    y = []
    j = 0
    s = lambda x, i: init_values[i+1] + b[i]*(x - init_args[i+1]) + c[i]*(x - init_args[i+1])**2 + d[i]*(x - init_args[i+1])**3
    for i in range(n - 1):
        # finds values of arguments in [x_{i-1}, x_i]
        while j < len(x) and init_args[i] <= x[j] <= init_args[i + 1]:
            y.append(s(x[j], i))
            j += 1
            
    return y


def ecubic(x, y) -> tuple[float, float]:
    """
    Absolute and relation error of cubic spline
    """

    aer, rer = 0, 0

    for i in range(len(x)):
        aer = max(aer, abs(y[i] - f(x[i])))
        rer = max(rer, abs((y[i] - f(x[i])) / f(x[i])))

    return round(aer, 7), round(rer * 100, 2)


def main():
    # f = lambda x: np.cos(x) * x**3 * np.sqrt(x)
    #init_args_1 = [-3, -1, 2, 3, 4]
    #init_values_1 = [f(x) for x in init_args_1]
    #init_args_1 = list(set([randint(0, 200) for _ in range(300)]))
    #init_values_1 = [f(x) for x in init_args_1]

    h = 0.01
    x1 = np.arange(min(init_args_1), max(init_args_1), h)
    y1 = cuspline(init_args_1, init_values_1, x1) 
    print(*ecubic(x1, y1))

    x2 = np.arange(min(init_args_2), max(init_args_2), h)
    y2 = cuspline(init_args_2, init_values_2, x1) 
    print(*ecubic(x2, y2))

    print_funcan_interpolate(x1, y1, init_args_1, init_values_1, func=f)
    print_funcan_interpolate(x2, y2, init_args_2, init_values_2, func=f)
    

if __name__ == "__main__":
    main()
