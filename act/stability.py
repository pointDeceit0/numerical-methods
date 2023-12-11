import matplotlib.pyplot as plt
import numpy as np


def mcriterion(w: list[float], root_number: int, f: str) -> tuple[list[float], list[float], bool]:
    """
    Mikhailov criterion.
        w:           np.arrange, defenition scope

        root_number: int, number of equations root, defines with degree of the polynomial
                          is used to define stability of the equation

        f: str, polynomial

    f has a specific format to find it with eval.
    """
    
    re, im = [0] * len(w), [0] * len(w)
    q = -1 # quadrant
    count = 0 
    for i, w0 in enumerate(w):
        x = complex(0, w0)
        v = eval(f)
        re[i] = v.real
        im[i] = v.imag
        
        # defines if curve wasn't in quadrant 
        if v.real > 0 and v.imag > 0: # 1
            if q != 0:
                count += 1
                q = 0  
        elif v.real < 0 and v.imag > 0: # 2
            if q != 1:
                count += 1
                q = 1
        elif v.real < 0 and v.imag < 0: # 3
            if q != 2:
                count += 1
                q = 2
        elif v.real > 0 and v.imag < 0: # 4
            if q != 3:
                count += 1
                q = 3

    return re, im, count == root_number


def main():
    with open("act/idata.txt", "r", encoding="utf-8") as file:
        # task number: number, equation, number of roots
        tasks = {}
        cur_task = ""
        while x := file.readline().strip().split():
            if len(x) == 1:
                tasks[x[0]] = []
                cur_task = x[0]
            else:
                tasks[cur_task].append(x)

    h = 0.005
    w = np.arange(0, 4, h)
    for k, data in tasks.items():

        fgr, ax = plt.subplots(nrows=1, ncols=len(data))
        for i, d in enumerate(data):
            re, im, stable = mcriterion(w, int(d[2]), d[1])

            ax[i].plot(re, im, "red")
            ax[i].grid(True)
            ax[i].set_title(f'Задание {k} {d[0]}')
            # axes
            ax[i].spines['left'].set_position(('data', 0))
            ax[i].spines['bottom'].set_position(('data', 0))
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].set_xlabel("Re(Q)", loc="right")
            ax[i].set_ylabel("Im(Q)", loc="top")

            if stable:
                print(f'Уравнение в задании {k} {d[0]} - устойчиво')
            else:
                print(f'Уравнение в задании {k} {d[0]} - не устойчиво')
        print()
        plt.show()
            

if __name__ == "__main__":
    main()