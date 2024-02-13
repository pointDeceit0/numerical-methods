import matplotlib.pyplot as plt
from numpy import arccos

def f(x):
    return arccos(x)


x0 = 0.1
init_args_1 = [-0.4, -0.1, 0.2, 0.5]
init_args_2 = [-0.4, 0, 0.2, 0.5]

init_values_1 = [f(x) for x in init_args_1]
init_values_2 = [f(x) for x in init_args_2]


def print_funcan(x: list[float], *y: list[list[float]], names: list[str], func=f) -> None:
    """
    Prints analytical and numerical plots 

    Analytical builds with inition function
    """

    for g in y:
        print(g)
        print(y)
        plt.plot(x, g, linewidth=4)   # numerical
    plt.plot(x, func(x)) # analytical
    if names:
        names.append("Analytical")
        plt.legend(names)

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


def print_funcan_interpolate(x: list[float], y: list[float], init_args: list[float], init_values: list[float], func=f) -> None:
    """
    Prints analytical and numerical plots, also interpollation nodes

    Analytical builds with inition function
    """

    plt.plot(x, y, linewidth=4)   # numerical
    plt.plot(x, func(x)) # analytical
    plt.plot(init_args, init_values, ".")
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


def wn(init_args: list[float], x0: float) -> float:
    '''
    Calculates value of wn

    wn(x) = mul(x0 - xj) for j = 1 to n
    '''
    wn = 1
    for x in init_args:
        wn *= abs(x0 - x)
    return wn