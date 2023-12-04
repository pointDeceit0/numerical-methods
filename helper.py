import matplotlib.pyplot as plt


def print_funcan(x, y, func=func) -> None:
    """
    Prints analytical and numerical plots 

    Analytical builds with inition function
    """

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