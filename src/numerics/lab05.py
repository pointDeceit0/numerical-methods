import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np  
from helper import *


def f(x: float) -> float:
    return x**3 * np.sqrt(x**2 - 49)


def intf(x: float) -> float:
    return (x**2 - 49)**(3/2) * (3*x**2 + 98) / 15


def print_integration(x: list[float], rects: list[Rectangle]) -> None:
    fig, ax = plt.subplots()
    x0 = np.arange(x[0], x[-1], 0.001)
    ax.plot(x0, f(x0), "blue", linewidth=2)
    ax.grid("both")
    
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("x", loc="right")
    ax.set_ylabel("y", loc="top")

    for r in rects:
        ax.add_patch(r)
    
    plt.show()


def irectan(x: list[float], f=f) -> tuple[float, float, float]:
    """
    Integration rectangular.
        x: list[float], scope of defenition

        f: intergrated function

        Return: (value of numeric integration, absolute error, relation error)
    """
    s = 0

    rectangles = []
    for i in range(1, len(x)):
        s += f((x[i] + x[i-1]) / 2) * (x[i] - x[i-1])
        rectangles.append(Rectangle((x[i-1], 0), x[i] - x[i-1], f((x[i] + x[i-1]) / 2),
                    edgecolor="red",
                    facecolor="pink",
                    lw=1
            )
        )

    print_integration(x, rectangles)

    num_s = intf(x[-1]) - intf(x[0])

    return round(s, 2), round(abs(s - num_s), 2), round(abs(s - num_s) / num_s * 100, 2)


def itrapez(x: list[float], f=f) -> tuple[float, float, float]:
    """
    Integration trapezoid.
        x: list[float], scope of defenition

        f: intergrated function

        Return: (value of numeric integration, absolute error, relation error)
    """
    s = 0

    polygons = []
    for i in range(1, len(x)):
        s += (f(x[i]) + f(x[i-1])) * (x[i] - x[i-1]) / 2
        polygons.append(Polygon([[x[i-1], 0], [x[i], 0], [x[i], f(x[i])], [x[i-1], f(x[i-1])]],
                    edgecolor="red",
                    facecolor="pink",
                    lw=1
            )
        )
        
    print_integration(x, polygons)

    num_s = intf(x[-1]) - intf(x[0])
    
    return round(s, 2), round(abs(s - num_s), 2), round(abs(s - num_s) / num_s * 100, 2)


def isimpson(x: list[float], f=f) -> tuple[float, float, float]:
    """
    Integration by Simpson method.
        x: list[float], scope of defenition

        f: intergrated function

        Return: (value of numeric integration, absolute error, relation error)
    """
    s = 0

    polygons = []
    for i in range(1, len(x)):
        s += (f(x[i-1]) + 4*f((x[i] + x[i-1]) / 2) + f(x[i])) * (x[i] - x[i-1]) / 6
        polygons.append(Polygon([[x[i-1], 0], [x[i], 0], [x[i], f(x[i])], [x[i-1], f(x[i-1])], [(x[i] - x[i-1]) / 2, f((x[i] - x[i-1]) / 2)]],
                    edgecolor="red",
                    facecolor="pink",
                    lw=1
            )
        )

    print_integration(x, polygons)

    num_s = intf(x[-1]) - intf(x[0])

    return round(s, 2), round(abs(s - num_s), 4), round(abs(s - num_s) / num_s * 100, 4)


def main():
    h1 = 0.5
    h2 = 0.25

    l = 7.5
    r = 15
    x1 = np.arange(l, r + h1, h1)
    x2 = np.arange(l, r + h2, h2)

    print(irectan(x1))
    print(irectan(x2))
    print(itrapez(x1))
    print(itrapez(x2))
    print(isimpson(x1))
    print(isimpson(x2))


if __name__ == "__main__":
    main()