from matplotlib import pyplot as plt
import numpy as np
import subprocess
import math
import re


def hilbert_res() -> None:
    proc = subprocess.Popen("Main.exe Hilbert 26 Out2.txt", shell=True)
    proc.wait()
    xs = []
    ys = []
    with open("./Out2.txt") as f:
        content = f.read()
        for line in content.split('\n'):
            if line == "":
                continue
            x, y = line.split()
            xs.append(int(x))
            ys.append(math.log(float(y)))
    plt.xlabel("rank")
    plt.ylabel("$\ln $cond$(\mathbf{H}_{n})_2$")
    plt.scatter(xs, ys)
    plt.show()


def solve_res(size: int, num_iter: int, interval: tuple) -> None:
    def file_dump(path: str) -> list:
        ret = []
        with open(path) as f:
            content = f.read()
            content = content.split('\n\n')
            for item in content:
                nums = item.split('\n')
                while '' in nums:
                    nums.remove('')
                if nums == []:
                    continue
                vec = np.array(list(map(float, nums)))
                ret.append(vec)
        return ret

    def print_loss(solutions: list, method_name: str, interval: tuple) -> None:
        correct = np.ones_like(solutions[0])
        losses = []
        for solution in solutions:
            loss = ((solution - correct) ** 2).sum()
            losses.append(loss)
        lw = '--' if method_name.startswith('SOR') else '-'
        plt.plot(list(range(1, len(losses) + 1))[interval[0]: interval[1]],
                 losses[interval[0]: interval[1]], label=method_name, linestyle=lw)

    # subprocess.Popen(
    #     f"Main.exe Jacobi {size} Out.txt {num_iter}", shell=True).wait()
    # arr = file_dump('Out.txt')
    # print_loss(arr, 'Jacobi', interval)

    subprocess.Popen(
        f"Main.exe GaussSeidel {size} Out.txt {num_iter}", shell=True).wait()
    arr = file_dump('Out.txt')
    print_loss(arr, 'Gauss-Seidel', interval)

    for omega in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]:
        subprocess.Popen(
            f"Main.exe SOR {size} Out.txt {num_iter} {omega}", shell=True).wait()
        arr = file_dump('Out.txt')
        print_loss(arr, f'SOR with $\omega = {omega}$', interval)

    subprocess.Popen(
        f"Main.exe PCG {size} Out.txt {num_iter}", shell=True).wait()
    arr = file_dump('Out.txt')
    print_loss(arr, 'PCG', interval)

    plt.legend()
    plt.show()


def test_methods(n: int, num_iter: int, global_dict: dict):
    def get_solution(path: str):
        with open(path) as f:
            content = f.read().split('\n\n')
        while '\n' in content:
            content.remove('\n')
        content = content[-1].split('\n')
        while '' in content:
            content.remove('')
        return np.array(list(map(float, content)))

    def get_err(solution: np.array):
        return sum((solution-np.ones_like(solution)) ** 2) / solution.shape[0]

    subprocess.Popen(
        f"Main.exe Gauss {n} Out.txt", shell=True).wait()
    solution = get_solution("Out.txt")
    print('Gauss', solution, get_err(solution))
    global_dict['Gauss'][n] = get_err(solution)

    subprocess.Popen(
        f"Main.exe Jacobi {n} Out.txt {num_iter}", shell=True).wait()
    solution = get_solution("Out.txt")
    print('Jacobi', solution, get_err(solution))
    global_dict['Jacobi'][n] = get_err(solution)

    subprocess.Popen(
        f"Main.exe GaussSeidel {n} Out.txt {num_iter}", shell=True).wait()
    solution = get_solution("Out.txt")
    print('GaussSeidel', solution, get_err(solution))
    global_dict['GaussSeidel'][n] = get_err(solution)

    for omega in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]:
        subprocess.Popen(
            f"Main.exe SOR {n} Out.txt {num_iter} {omega}", shell=True).wait()
        solution = get_solution("Out.txt")
        print(f'SOR omega = {omega}', solution, get_err(solution))
        global_dict[f'SOR omega = {omega}'][n] = get_err(solution)

    subprocess.Popen(
        f"Main.exe PCG {n} Out.txt {num_iter}", shell=True).wait()
    solution = get_solution("Out.txt")
    print('PCG', solution, get_err(solution))
    global_dict['PCG'][n] = get_err(solution)


if __name__ == '__main__':
    # hilbert_res()
    # solve_res(200, 300, (0, 300))

    global_dict = {}
    global_dict['Gauss'] = {}
    global_dict['Jacobi'] = {}
    global_dict['GaussSeidel'] = {}
    for omega in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]:
        global_dict[f'SOR omega = {omega}'] = {}
    global_dict['PCG'] = {}

    for n in range(10, 101, 10):
        print(f"\n================\n{n}\n================\n")
        test_methods(n, n/2, global_dict)

    print("\n\n\n")
    for k in global_dict.keys():
        print(f'|{k}|{"|".join(map(lambda x: "% .2e" % x, global_dict[k].values()))}|')
