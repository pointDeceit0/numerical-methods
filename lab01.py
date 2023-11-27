import matplotlib.pyplot as plt
import numpy as np  
import random
from scipy.interpolate import lagrange 


def print_func(x, y) -> None:
    plt.plot(x, y, linewidth=4)   # numerical
    plt.plot(x, func(x)) # analytical
    plt.legend(["numerical", "analytical"])

    plt.grid(axis="both")
    # axes
    ax = plt.subplot()
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("x", loc="right")
    ax.set_ylabel("y", loc="top")

    plt.show()


def func(x: float):
    return np.arccos(x)


def interpolation_lagrange(init_args: list[float], init_values: list[float], interpol_args: list[float]) -> list[float]:
    y = []
    for x in interpol_args:
        res = 0
        for i, val1 in enumerate(init_args):
            cur_res = init_values[i]
            for j, val2 in enumerate(init_args):
                if i == j:
                    continue
                cur_res *= (x - val2) / (val1 - val2)
            res += cur_res
        y.append(res)

    return y


def task_1(init_args=[]) -> None:
    if len(init_args) == 0:
        init_args = [-0.963, -0.12, 0.67, 1]
    # build init values
    init_values = [func(v) for v in init_args] 

    h = 0.01
    x = np.arange(-1, 1, h) 
    y = interpolation_lagrange(init_args, init_values, x)

    # numerical-analytical relative error 
    dev = 0
    max_y = max([abs(func(x0)) for x0 in x])
    for i in range(len(x)):
        dev = max(dev, abs(func(x[i]) - y[i]) / max_y)
    print(round(dev, 3))

    print_func(x, y)
    

def task_2() -> None:
    X = [-0.4, 0, 0.2, 0.5]
    task_1(X)


def task_3(h) -> float:
    # generate inition array
    x =  - random.randint(58, 100) / 100
    init_args = []
    while x <= 1: 
        init_args.append(x)
        x += 0.3
    dev = 0
    init_values = [func(v) for v in init_args]
    x = np.linspace(min(init_args), max(init_args), 10)
    y = interpolation_lagrange(init_args, init_values, x)
    max_y = max([abs(func(x0)) for x0 in x])
    for i in range(len(x)):
        dev = max(dev, abs(func(x[i]) - y[i]) / max_y)

    print(round(dev, 3))


def main():
    #a = [-164, -56.36, -53, -32, -2, 0, 4, 47.36, 67, 98, 111, 120]
    task_1()
    task_2()
    task_3(0.1)

    #init_args = np.array(a)
    #poly = lagrange(init_args, func(init_args))
    #x = np.arange(min(init_args) - 1, max(init_args) + 1, 0.1)
    #y = [poly(v) for v in x]
#
    #dev = 0
    #max_y = max([abs(func(x0)) for x0 in x])
    #for i in range(len(x)):
    #    dev = max(dev, abs(func(x[i]) - y[i]) / max_y)
    #print(round(dev, 3))
#
    #print_func(x, y)




if __name__ == "__main__":
    main()






